import streamlit as st
from sentence_transformers import SentenceTransformer
import chromadb
import re

# Page config
st.set_page_config(page_title="Nutrition Advisor", page_icon="🥗", layout="wide")

# Title
st.title("🥗 Nutrition Advisor - Semantic Search")
st.markdown("Search through nutrition data using natural language queries")

# Initialize session state
if 'search_triggered' not in st.session_state:
    st.session_state.search_triggered = False
if 'current_query' not in st.session_state:
    st.session_state.current_query = ""
if 'current_n_results' not in st.session_state:
    st.session_state.current_n_results = 10
if 'last_results' not in st.session_state:
    st.session_state.last_results = []

# Initialize ChromaDB client and collection
@st.cache_resource
def init_chromadb():
    client = chromadb.PersistentClient(path="./chroma_data")
    try:
        collection = client.get_collection("nutrition_data")
    except:
        st.error("Collection 'nutrition_data' not found. Please run the data loading script first.")
        st.stop()
    return collection

collection = init_chromadb()

# Initialize embedding model
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_model()

# Function to detect query intent
def analyze_query_intent(query):
    """
    Detect if user wants to avoid/find low amounts of something or find high amounts
    Returns: (modified_query, is_avoidance, nutrient_term)
    """
    query_lower = query.lower()
    
    # Avoidance/negation keywords
    avoidance_keywords = [
        'cannot eat', 'can\'t eat', 'should not eat', 'shouldn\'t eat',
        'avoid', 'low in', 'reduce', 'limit', 'restrict',
        'without', 'no ', 'less ', 'minimal', 'decrease'
    ]
    
    # High/seeking keywords
    seeking_keywords = [
        'high in', 'rich in', 'lots of', 'plenty of', 'more ',
        'increase', 'boost', 'good source', 'need more'
    ]
    
    # Check for avoidance intent
    is_avoidance = any(keyword in query_lower for keyword in avoidance_keywords)
    is_seeking = any(keyword in query_lower for keyword in seeking_keywords)
    
    # Extract nutrient/food term
    nutrient_patterns = [
        r'(iron|protein|vitamin|calcium|sodium|sugar|carb|fat|calorie|fiber)',
        r'(vitamin [a-k]|vitamin b\d+)',
    ]
    
    nutrient_term = None
    for pattern in nutrient_patterns:
        match = re.search(pattern, query_lower)
        if match:
            nutrient_term = match.group(1)
            break
    
    # Create modified query for embedding
    if is_avoidance and nutrient_term:
        # Search for LOW nutrient foods
        modified_query = f"low {nutrient_term} foods"
        search_mode = "avoidance"
    elif is_seeking and nutrient_term:
        # Search for HIGH nutrient foods
        modified_query = f"high {nutrient_term} foods"
        search_mode = "seeking"
    else:
        # Use original query
        modified_query = query
        search_mode = "neutral"
    
    return modified_query, search_mode, nutrient_term

# Search interface
st.subheader("🔍 Search Nutrition Data")

query = st.text_input(
    "Enter your nutrition question:",
    placeholder="e.g., high protein foods, I cannot eat high iron food, low calorie snacks"
)

col1, col2 = st.columns([3, 1])
with col1:
    n_results = st.slider("Number of results", min_value=1, max_value=20, value=10)
with col2:
    similarity_threshold = st.slider("Relevance threshold", min_value=0.0, max_value=1.0, value=0.3, step=0.1, help="Higher = stricter matching")

# Trigger search on button click OR when slider changes (if previous search exists)
search_clicked = st.button("Search", type="primary")

if search_clicked and query:
    st.session_state.current_query = query
    st.session_state.current_n_results = n_results
    st.session_state.search_triggered = True
elif st.session_state.current_query and n_results != st.session_state.current_n_results:
    # Slider changed - auto-refresh results
    st.session_state.current_n_results = n_results
    st.session_state.search_triggered = True

# Perform search if triggered
if st.session_state.search_triggered and st.session_state.current_query:
    with st.spinner("🔍 Analyzing your query and searching..."):
        try:
            # Analyze query intent
            modified_query, search_mode, nutrient_term = analyze_query_intent(st.session_state.current_query)
            
            # Show intent detection
            if search_mode == "avoidance":
                st.info(f"🔍 Detected: You want to **avoid/reduce {nutrient_term}**. Searching for LOW {nutrient_term} foods...")
            elif search_mode == "seeking":
                st.info(f"🔍 Detected: You want **high {nutrient_term}**. Searching for foods rich in {nutrient_term}...")
            
            # Generate query embedding using modified query
            query_embedding = model.encode([modified_query])[0].tolist()
            
            # Search ChromaDB - get more results than needed for filtering
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=min(50, st.session_state.current_n_results * 3)
            )
            
            # Extract query keywords for boosting
            query_words = set(modified_query.lower().split())
            
            # Score and filter results
            scored_results = []
            for i in range(len(results['metadatas'][0])):
                metadata = results['metadatas'][0][i]
                distance = results['distances'][0][i]
                
                # Check if query words appear in food name or description
                food_text = f"{metadata['Food']} {metadata['Description']}".lower()
                keyword_matches = sum(1 for word in query_words if word in food_text)
                
                # Additional filtering for avoidance queries
                if search_mode == "avoidance" and nutrient_term:
                    # Check if the nutrient appears HIGH in the nutrient data
                    nutrient_data_lower = metadata['Nutrient Data'].lower()
                    # If searching for low iron, penalize foods that mention high iron amounts
                    if nutrient_term in nutrient_data_lower:
                        # Extract number before nutrient term if possible
                        pattern = rf'(\d+\.?\d*)\s*(?:mg|g|mcg).*{nutrient_term}'
                        match = re.search(pattern, nutrient_data_lower)
                        if match:
                            amount = float(match.group(1))
                            # Penalize high amounts when avoiding
                            if amount > 5:  # Threshold for "high"
                                distance += 0.3  # Increase distance (lower similarity)
                
                # Boost score if keywords match
                adjusted_distance = distance - (keyword_matches * 0.15)
                similarity = 1 - distance
                
                # Apply similarity threshold
                if similarity >= similarity_threshold:
                    scored_results.append({
                        'metadata': metadata,
                        'distance': adjusted_distance,
                        'original_distance': distance,
                        'similarity': similarity,
                        'keyword_matches': keyword_matches
                    })
            
            # Sort by adjusted distance (lower is better)
            scored_results.sort(key=lambda x: x['distance'])
            
            # Take top n_results
            final_results = scored_results[:st.session_state.current_n_results]
            
            # Store for AI enhancement
            st.session_state.last_results = [r['metadata'] for r in final_results]
            
            # Display results
            if final_results:
                st.success(f"✅ Found {len(final_results)} relevant results!")
                
                st.markdown("### 📊 Search Results")
                
                for i, result in enumerate(final_results):
                    metadata = result['metadata']
                    similarity = result['similarity']
                    keyword_matches = result['keyword_matches']
                    
                    # Create title with match indicator
                    match_emoji = "🎯" if keyword_matches > 0 else "📍"
                    title = f"{match_emoji} **{i+1}. {metadata['Food']}** (Match: {similarity:.1%})"
                    
                    with st.expander(title, expanded=(i==0)):
                        col1, col2 = st.columns([1, 2])
                        
                        with col1:
                            st.markdown("**Category:**")
                            st.write(metadata['Category'])
                        
                        with col2:
                            st.markdown("**Description:**")
                            st.write(metadata['Description'])
                        
                        st.markdown("**Nutrient Data:**")
                        st.write(metadata['Nutrient Data'])
                        
                        # Show match details
                        if keyword_matches > 0:
                            st.caption(f"🎯 {keyword_matches} keyword match(es)")
            else:
                st.warning(f"⚠️ No results found with similarity ≥ {similarity_threshold:.0%}. Try lowering the relevance threshold or using different keywords.")
                
        except Exception as e:
            st.error(f"❌ Search error: {str(e)}")
            import traceback
            st.code(traceback.format_exc())
elif search_clicked and not query:
    st.warning("⚠️ Please enter a search query")

# Sidebar info
with st.sidebar:
    st.header("ℹ️ About")
    st.write("""
    This app uses semantic search with intent detection to find relevant nutrition information.
    
    **How it works:**
    1. Enter a natural language query
    2. The app detects if you want high/low nutrients
    3. Searches and ranks results accordingly
    
    **Powered by:**
    - Sentence Transformers
    - ChromaDB
    """)
    
    st.header("💡 Tips")
    st.markdown("""
    - 🎯 = Exact keyword match
    - 📍 = Semantic match
    - Say "cannot eat" or "avoid" for low amounts
    - Say "high in" or "rich in" for high amounts
    - The app detects your intent automatically
    """)
    
    st.header("📝 Example Queries")
    st.markdown("""
    **Avoidance:**
    - "I cannot eat high iron food"
    - "foods low in sodium"
    - "avoid sugar"
    
    **Seeking:**
    - "high protein foods"
    - "rich in vitamin C"
    - "good source of calcium"
    """)