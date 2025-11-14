import streamlit as st
import chromadb
from sentence_transformers import SentenceTransformer
import re
import json
import os

# Initialize ChromaDB client
client = chromadb.PersistentClient(path="./chromadb_data")

try:
    collection = client.get_collection(name="nutrition_foods")
except:
    st.error("Collection 'nutrition_foods' not found. Please run the data loading script first.")
    st.stop()

# Load sentence transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Load nutrients data
if os.path.exists('nutrients_data.json'):
    with open('nutrients_data.json', 'r') as f:
        nutrients_data = json.load(f)
else:
    st.error("nutrients_data.json not found. Please run the data loading script first.")
    st.stop()

# ==================== CONFIGURATION ====================

NUTRIENT_MAPPING = {
    'protein': 'protein', 'fat': 'fat', 'carbs': 'carbs', 'carbohydrates': 'carbs',
    'fiber': 'fiber', 'fibre': 'fiber', 'sugar': 'sugar', 'sodium': 'sodium',
    'salt': 'sodium', 'calcium': 'calcium', 'iron': 'iron', 'cholesterol': 'cholesterol',
    'calories': 'calories', 'vitamin_a': 'vitamin_a', 'vitamina': 'vitamin_a',
    'vitamin_c': 'vitamin_c', 'vitaminc': 'vitamin_c',
}

# Nutrient scoring thresholds: {nutrient: [(threshold, score), ...]}
SEEK_THRESHOLDS = {
    'protein': [(30, 30), (25, 25), (20, 20), (15, 15), (10, 10), (5, 5)],
    'fiber': [(30, 30), (25, 25), (20, 20), (15, 15), (10, 10), (5, 5)],
    'calcium': [(500, 30), (300, 20), (150, 10)],
    'iron': [(10, 30), (5, 20), (3, 10)],
    'vitamin_c': [(50, 30), (25, 20), (10, 10), (5, 5)],
    'vitamin_a': [(50, 30), (25, 20), (10, 10), (5, 5)],
}

AVOID_THRESHOLDS = {
    'fat': [(30, -25), (20, -20), (15, -15), (10, -10), (5, 15), (8, 10)],
    'sugar': [(30, -25), (20, -20), (15, -15), (10, -10), (5, 15), (8, 10)],
    'sodium': [(500, -25), (300, -15), (150, -10), (100, 15)],
    'cholesterol': [(100, -20), (50, -10), (20, 10)],
    'iron': [(0.5, 25), (1.0, 20), (2.0, 15), (3.0, 5), (5, -15), (3, -5)],
    'carbs': [(30, -20), (20, -10), (10, 15)],
}

SEEK_PATTERNS = [
    r'high\s+(?:in\s+)?(\w+)(?:\s+but|\s+and|\s+with|,|\s|$)',
    r'rich\s+in\s+(\w+)(?:\s+but|\s+and|\s+with|,|\s|$)',
    r'good\s+source\s+of\s+(\w+)(?:\s+but|\s+and|\s+with|,|\s|$)',
    r'lots\s+of\s+(\w+)(?:\s+but|\s+and|\s+with|,|\s|$)',
    r'plenty\s+of\s+(\w+)(?:\s+but|\s+and|\s+with|,|\s|$)'
]

AVOID_PATTERNS = [
    r'but\s+low\s+(?:in\s+)?(\w+(?:\s+\w+)?)',
    r'low\s+(?:in\s+)?(\w+(?:\s+\w+)?)',
    r'avoid\s+(\w+(?:\s+\w+)?)',
    r'reduce\s+(\w+(?:\s+\w+)?)',
    r'without\s+(\w+(?:\s+\w+)?)',
    r'no\s+(\w+(?:\s+\w+)?)'
]

MEAT_KEYWORDS = [
    'beef', 'pork', 'chicken', 'turkey', 'lamb', 'veal', 'meat', 
    'fish', 'seafood', 'bacon', 'ham', 'sausage', 'salami', 'duck',
    'goose', 'game', 'venison', 'bison', 'steak', 'ribs', 'chop'
]

DAIRY_KEYWORDS = ['cheese', 'milk', 'egg', 'dairy', 'yogurt', 'butter', 'cream', 'whey']

VEGETABLE_KEYWORDS = [
    'vegetable', 'lettuce', 'tomato', 'carrot', 'broccoli', 'spinach', 
    'kale', 'potato', 'onion', 'pepper', 'cucumber', 'celery', 'cabbage',
    'cauliflower', 'zucchini', 'squash', 'eggplant', 'mushroom', 'corn',
    'peas', 'beans', 'asparagus', 'beet', 'radish', 'turnip'
]

# ==================== HELPER FUNCTIONS ====================

def normalize_nutrient_name(nutrient):
    """Normalize captured nutrient names to match data keys"""
    clean = nutrient.replace(' ', '').replace('_', '').lower()
    return NUTRIENT_MAPPING.get(clean, None)


def apply_threshold_scoring(value, thresholds, ascending=True):
    """Apply threshold-based scoring logic"""
    score = 0
    for threshold, points in thresholds:
        if ascending:
            if value > threshold:
                score = points
                break
        else:
            if value < threshold:
                score = points
                break
    return score


def preprocess_query(query):
    """Extract intent and constraints from query"""
    intent = {
        'avoid_nutrients': [],
        'seek_nutrients': [],
        'modifiers': [],
        'original_query': query
    }
    
    query_lower = query.lower()
    
    # Detect avoid and seek patterns
    for pattern in AVOID_PATTERNS:
        intent['avoid_nutrients'].extend(re.findall(pattern, query_lower))
    
    for pattern in SEEK_PATTERNS:
        intent['seek_nutrients'].extend(re.findall(pattern, query_lower))
    
    # Normalize and deduplicate
    intent['avoid_nutrients'] = list(set(filter(None, [normalize_nutrient_name(n) for n in intent['avoid_nutrients']])))
    intent['seek_nutrients'] = list(set(filter(None, [normalize_nutrient_name(n) for n in intent['seek_nutrients']])))
    
    # Detect modifiers
    modifiers = ['healthy', 'lean', 'fresh', 'organic', 'natural', 'whole', 'clean']
    intent['modifiers'] = [mod for mod in modifiers if mod in query_lower]
    
    return intent


def calculate_similarity_percentage(distance):
    """Convert ChromaDB cosine distance to percentage"""
    return max(0, min(100, round((1 - (distance / 2)) * 100, 2)))


def calculate_nutritional_match(query, nutrients, intent):
    """Calculate nutritional alignment score based on query intent"""
    score = 50
    
    # Boost for seek nutrients
    for nutrient in intent['seek_nutrients']:
        value = nutrients.get(nutrient, 0)
        if nutrient in SEEK_THRESHOLDS:
            for threshold, points in SEEK_THRESHOLDS[nutrient]:
                if value > threshold:
                    score += points
                    break
    
    # Penalty for avoid nutrients
    for nutrient in intent['avoid_nutrients']:
        value = nutrients.get(nutrient, 0)
        if nutrient in AVOID_THRESHOLDS:
            for threshold, points in AVOID_THRESHOLDS[nutrient]:
                if nutrient == 'iron':
                    # Iron avoidance uses different logic
                    if value < 0.5:
                        score += 25
                        break
                    elif value < 1.0:
                        score += 20
                        break
                    elif value < 2.0:
                        score += 15
                        break
                    elif value < 3.0:
                        score += 5
                        break
                    elif value > 5:
                        score -= 15
                        break
                    elif value > 3:
                        score -= 5
                        break
                else:
                    if value > threshold:
                        score += points
                        break
                    elif value < threshold and points > 0:
                        score += points
                        break
    
    # Apply modifiers
    if 'healthy' in intent['modifiers']:
        if nutrients.get('fat', 100) < 8 and nutrients.get('sodium', 1000) < 300 and nutrients.get('protein', 0) > 10:
            score += 25
        elif nutrients.get('fat', 100) < 12 and nutrients.get('sodium', 1000) < 400:
            score += 15
    
    if 'lean' in intent['modifiers']:
        if nutrients.get('fat', 100) < 5 and nutrients.get('protein', 0) > 18:
            score += 25
        elif nutrients.get('fat', 100) < 8 and nutrients.get('protein', 0) > 12:
            score += 15
    
    return min(100, max(0, score))


def apply_nutritional_filters(results, query, nutrients_data, intent):
    """Filter results based on nutritional criteria"""
    filters = {}
    query_lower = query.lower()
    exclude_keywords = []
    
    # Check for dietary restrictions
    if any(word in query_lower for word in ['vegetarian', 'vegan', 'plant-based', 'plant based', 'meatless', 'meat-free']):
        exclude_keywords = MEAT_KEYWORDS.copy()
        if 'vegan' in query_lower:
            exclude_keywords.extend(DAIRY_KEYWORDS)
    
    # Query-based filters
    filter_mapping = {
        'healthy': {'fat': (0, 15), 'sodium': (0, 500)},
        'lean': {'fat': (0, 8), 'protein': (12, 1000)},
        'low fat': {'fat': (0, 3)},
        'lowfat': {'fat': (0, 3)},
        'low sodium': {'sodium': (0, 140)},
        'low sugar': {'sugar': (0, 5)},
        'low calorie': {'calories': (0, 100)},
    }
    
    for keyword, nutrient_filters in filter_mapping.items():
        if keyword in query_lower:
            filters.update(nutrient_filters)
    
    # Intent-based filters
    seek_filter_map = {'protein': (20, 1000), 'fiber': (3, 1000), 'calcium': (100, 10000), 'iron': (3, 1000)}
    avoid_filter_map = {'fat': (0, 10), 'sodium': (0, 300), 'sugar': (0, 8), 'iron': (0, 3), 'cholesterol': (0, 50), 'carbs': (0, 15)}
    
    for nutrient in intent['seek_nutrients']:
        if nutrient in seek_filter_map:
            filters[nutrient] = seek_filter_map[nutrient]
    
    for nutrient in intent['avoid_nutrients']:
        if nutrient in avoid_filter_map:
            filters[nutrient] = avoid_filter_map[nutrient]
    
    # If no filters, return as-is
    if not filters and not exclude_keywords:
        return results
    
    # Apply filters
    filtered = []
    for result in results:
        food_id = result.get('id')
        if not food_id or food_id not in nutrients_data:
            continue
        
        # Check exclusions
        description = result['metadata'].get('description', '').lower()
        category = result['metadata'].get('category', '').lower()
        
        if any(keyword in description or keyword in category for keyword in exclude_keywords):
            continue
        
        # Check nutrient filters
        nutrients = nutrients_data[food_id].get('nutrients', {})
        if all(min_val <= nutrients.get(nutrient, 0) <= max_val for nutrient, (min_val, max_val) in filters.items()):
            filtered.append(result)
    
    return filtered if len(filtered) >= 3 else results[:15]


# ==================== MAIN SEARCH FUNCTION ====================

def perform_search(query, n_results=10, threshold=0.5):
    """Main search function with all enhancements"""
    intent = preprocess_query(query)
    nutrients_specified = bool(intent['seek_nutrients'] or intent['avoid_nutrients'] or intent['modifiers'])
    
    raw_results = collection.query(query_texts=[query], n_results=min(n_results * 3, 50))
    
    processed_results = []
    for i in range(len(raw_results['ids'][0])):
        food_id = raw_results['ids'][0][i]
        distance = raw_results['distances'][0][i]
        metadata = raw_results['metadatas'][0][i]
        
        food_data = nutrients_data.get(food_id, {})
        nutrients = food_data.get('nutrients', {})
        
        semantic_score = calculate_similarity_percentage(distance)
        
        if nutrients_specified:
            nutritional_score = calculate_nutritional_match(query, nutrients, intent)
            combined_score = (semantic_score * 0.4) + (nutritional_score * 0.6)
        else:
            nutritional_score = None
            combined_score = semantic_score
        
        processed_results.append({
            'id': food_id,
            'distance': distance,
            'metadata': metadata,
            'nutrients': nutrients,
            'semantic_score': semantic_score,
            'nutritional_score': nutritional_score,
            'combined_score': combined_score
        })
    
    filtered_results = apply_nutritional_filters(processed_results, query, nutrients_data, intent)
    filtered_results.sort(key=lambda x: x['combined_score'], reverse=True)
    final_results = [r for r in filtered_results if r['combined_score'] >= (threshold * 100)]
    
    return final_results[:n_results], intent, nutrients_specified


# ==================== STREAMLIT UI ====================

st.set_page_config(page_title="Nutrition Advisor", page_icon="🥗", layout="wide")

tab1, tab2 = st.tabs(["🔍 Search", "ℹ️ About"])

with tab1:
    st.title("🥗 Nutrition Advisor - Semantic Search")
    st.markdown("Search nutrition data using natural language")

    with st.sidebar:
        st.header("ℹ️ About")
        st.write("Semantic search with automatic intent detection for nutrition data.")
        
        st.header("💡 Tips")
        st.write('• Use "high in" or "rich in" for nutrients you want')
        st.write('• Use "low" or "avoid" for nutrients to minimize')
        st.write('• Combine requirements: "high protein but low fat"')
        
        st.header("Example Queries")
        st.write('• "high protein foods"')
        st.write('• "rich in vitamin C but low sugar"')
        st.write('• "lean healthy meat"')
        st.write('• "vegetarian high calcium"')

    query = st.text_input("Enter your nutrition question:", placeholder="e.g., high in protein but low in iron")

    col1, col2 = st.columns([2, 1])
    with col1:
        n_results = st.slider("Number of results:", 1, 20, 10)
    with col2:
        threshold = st.slider("Relevance threshold:", 0.0, 1.0, 0.0, 0.05)

    if st.button("🔍 Search", type="primary"):
        if query:
            with st.spinner("Searching..."):
                results, intent, nutrients_specified = perform_search(query, n_results, threshold)
                
                # Display detected intent
                if intent['seek_nutrients'] or intent['avoid_nutrients'] or intent['modifiers']:
                    st.markdown("---")
                    st.subheader("🎯 Detected Intent:")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    if intent['seek_nutrients']:
                        col1.success(f"**WANT HIGH:** {', '.join(intent['seek_nutrients'])}")
                    if intent['avoid_nutrients']:
                        col2.warning(f"**AVOID/LOW:** {', '.join(intent['avoid_nutrients'])}")
                    if intent['modifiers']:
                        col3.info(f"**Modifiers:** {', '.join(intent['modifiers'])}")
                
                # Display scoring method
                st.markdown("---")
                if nutrients_specified:
                    st.info("📊 **Scoring**: 40% semantic + 60% nutritional match")
                else:
                    st.info("📊 **Scoring**: 100% semantic similarity")
                
                st.markdown("---")
                if results:
                    st.success(f"✅ Found {len(results)} results")
                    st.subheader("📊 Search Results")
                    st.caption("*All nutrient values shown are per 100g of food")
                    
                    for i, result in enumerate(results):
                        with st.expander(f"**{i+1}. {result['metadata'].get('description', 'Unknown')}**", expanded=(i<3)):
                            col1, col2, col3 = st.columns(3)
                            col1.metric("Semantic Match", f"{result['semantic_score']:.1f}%")
                            
                            if result['nutritional_score'] is not None:
                                col2.metric("Nutritional Match", f"{result['nutritional_score']:.1f}%")
                            else:
                                col2.metric("Nutritional Match", "N/A")
                            
                            col3.metric("Combined Score", f"{result['combined_score']:.1f}%")
                            
                            st.markdown(f"**Category:** {result['metadata'].get('category', 'Unknown')}")
                            
                            nutrients = result['nutrients']
                            if nutrients:
                                st.markdown("**Nutrients (per 100g):**")
                                nutrient_text = ", ".join([f"{k}: {v}{'kcal' if k == 'calories' else 'g'}" 
                                                          for k, v in nutrients.items()])
                                st.text(nutrient_text)
                            
                            # Highlight matches
                            if intent['seek_nutrients'] or intent['avoid_nutrients']:
                                matches = []
                                for nutrient in intent['seek_nutrients']:
                                    if nutrient in nutrients and nutrients[nutrient] > 15:
                                        matches.append(f"✅ High {nutrient}: {nutrients[nutrient]}g")
                                for nutrient in intent['avoid_nutrients']:
                                    if nutrient in nutrients and nutrients[nutrient] < 5:
                                        matches.append(f"✅ Low {nutrient}: {nutrients[nutrient]}g")
                                
                                if matches:
                                    st.success(" | ".join(matches))
                else:
                    st.warning("No results found. Try adjusting threshold or query.")
        else:
            st.error("Please enter a search query")

with tab2:
    st.title("ℹ️ About This App")
    
    st.header("📚 Data Source")
    st.write("""
    **USDA FoodData Central** - FNDDS Survey Foods (October 2024)
    
    - ~5,400 foods commonly consumed in the US
    - Filtered for quality (removed baby foods, supplements, incomplete data)
    - 12 key nutrients tracked per food
    - **All nutrient values are per 100g of food**
    - [Download source](https://fdc.nal.usda.gov/download-datasets.html)
    """)
    
    st.header("🔧 How It Works")
    st.write("""
    **1. Intent Detection**: Automatically understands what you want
    - Detects "high in", "low", "avoid" patterns
    - Identifies modifiers like "healthy", "lean"
    - Recognizes dietary restrictions (vegetarian, vegan)
    
    **2. Semantic Search**: Uses AI to understand meaning, not just keywords
    - Converts your query to a vector embedding
    - Searches 5,400+ foods in ChromaDB vector database
    - Model: `all-MiniLM-L6-v2`
    
    **3. Adaptive Scoring**:
    - **No nutrient requirements**: 100% semantic similarity
    - **With nutrient requirements**: 40% semantic + 60% nutritional match
    
    **4. Smart Filtering**: Applies filters based on detected intent
    - Nutrient thresholds (high/low values)
    - Dietary restrictions (excludes meat/dairy)
    - Health modifiers ("healthy", "lean")
    """)
    
    st.header("💡 Usage Tips")
    st.write("""
    **Express what you want:**
    - "high protein foods", "rich in calcium"
    
    **Express what to avoid:**
    - "low fat", "avoid sugar", "no iron"
    
    **Combine requirements:**
    - "high protein but low fat"
    - "healthy breakfast with fiber"
    
    **Use modifiers:**
    - healthy, lean, fresh, vegetarian, vegan
    
    **Adjust search parameters:**
    - Number of results (1-20)
    - Relevance threshold (0-1.0)
    """)
    
    st.header("🛠️ Tech Stack")
    st.write("""
    - **UI**: Streamlit
    - **Embeddings**: Sentence-Transformers (all-MiniLM-L6-v2)
    - **Vector DB**: ChromaDB
    - **Data**: USDA FoodData Central (Oct 2024)
    - **Intent Detection**: Regex pattern matching
    """)
    
    st.markdown("---")
    st.markdown("*Data last updated: October 2024*")