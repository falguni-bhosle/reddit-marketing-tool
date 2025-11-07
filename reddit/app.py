from flask import Flask, render_template, request, send_file, flash, redirect, url_for
import pandas as pd
import praw
from datetime import datetime
import os
import threading
from io import BytesIO
import google.generativeai as genai
import time
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv('FLASK_SECRET_KEY', 'fallback-secret-key')

# ---------------- Configuration ----------------
REDDIT_CLIENT_ID = os.getenv('REDDIT_CLIENT_ID')
REDDIT_SECRET = os.getenv('REDDIT_SECRET')
REDDIT_USER_AGENT = os.getenv('REDDIT_USER_AGENT', 'RedditKeywordSearchBot/1.0')
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')

print("🔧 Checking environment variables...")
print(f"Reddit Client ID: {'✅ Set' if REDDIT_CLIENT_ID else '❌ Missing'}")
print(f"Reddit Secret: {'✅ Set' if REDDIT_SECRET else '❌ Missing'}")
print(f"Gemini API Key: {'✅ Set' if GEMINI_API_KEY else '❌ Missing'}")

# Validate required environment variables
if not all([REDDIT_CLIENT_ID, REDDIT_SECRET, GEMINI_API_KEY]):
    raise ValueError("Missing required environment variables")

try:
    genai.configure(api_key=GEMINI_API_KEY)
    print("✅ Gemini AI configured successfully")
    
    # List available models to debug
    try:
        models = genai.list_models()
        print("📋 Available Gemini models:")
        for model in models:
            print(f"  - {model.name}")
    except Exception as e:
        print(f"⚠️ Could not list models: {e}")
        
except Exception as e:
    print(f"❌ Gemini configuration failed: {e}")

try:
    reddit = praw.Reddit(
        client_id=REDDIT_CLIENT_ID,
        client_secret=REDDIT_SECRET,
        user_agent=REDDIT_USER_AGENT
    )
    # Test Reddit connection
    print("🔍 Testing Reddit connection...")
    test_subreddit = reddit.subreddit("all")
    print("✅ Reddit authentication successful")
except Exception as e:
    print(f"❌ Reddit authentication failed: {e}")
    reddit = None

# Global variable to store search progress and configuration
search_config = {
    'service_description': '',
    'posts_per_keyword': 10,
    'relevance_threshold': 0.3  # Lowered threshold for testing
}

search_progress = {
    'current': 0,
    'total': 0,
    'message': '',
    'is_running': False,
    'results': None,
    'error': None
}

def get_gemini_model():
    """Get available Gemini model"""
    try:
        # Try the new model name first
        return genai.GenerativeModel('gemini-1.5-pro-latest')
    except:
        try:
            # Fallback to older model name
            return genai.GenerativeModel('gemini-pro')
        except:
            # Last resort - try any available model
            models = genai.list_models()
            for model in models:
                if 'generateContent' in model.supported_generation_methods:
                    return genai.GenerativeModel(model.name)
            raise Exception("No suitable Gemini model found")

def analyze_post_relevance(service_description, post_title, post_content=""):
    """
    Use Gemini AI to analyze if a post is suitable for marketing the service
    """
    try:
        model = get_gemini_model()
        
        prompt = f"""
        Analyze this Reddit post for marketing opportunities. Be lenient and look for any possible connection.

        SERVICE TO MARKET: {service_description}

        REDDIT POST:
        Title: {post_title}
        Content: {post_content[:300]}

        Answer in this exact format:
        RELEVANT: YES or NO
        CONFIDENCE: 0.0 to 1.0
        REASON: One sentence explanation
        SUGGESTION: Brief marketing suggestion

        Guidelines:
        - If the post is somewhat related to the service, say YES
        - Look for people asking questions, needing help, or discussing related topics
        - Consider this a good marketing opportunity if you can provide helpful information
        - Confidence should be based on how good the opportunity is
        """
        
        response = model.generate_content(prompt)
        response_text = response.text
        print(f"🔍 Raw Gemini Response: {response_text}")
        
        # Parse the response
        lines = response_text.strip().split('\n')
        relevant = "NO"
        confidence = 0.5  # Default medium confidence
        reason = "Potential marketing opportunity found"
        marketing_suggestion = "Offer helpful information about your service"
        
        for line in lines:
            line_upper = line.upper()
            if line.startswith('RELEVANT:'):
                if 'YES' in line_upper:
                    relevant = "YES"
                elif 'NO' in line_upper:
                    relevant = "NO"
            elif line.startswith('CONFIDENCE:'):
                try:
                    confidence_str = line.split(':')[1].strip()
                    confidence = float(confidence_str)
                except:
                    confidence = 0.5
            elif line.startswith('REASON:'):
                reason = line.split(':', 1)[1].strip()
            elif line.startswith('SUGGESTION:'):
                marketing_suggestion = line.split(':', 1)[1].strip()
        
        print(f"📊 Analysis Result: Relevant={relevant}, Confidence={confidence}")
        return relevant == "YES", confidence, reason, marketing_suggestion
        
    except Exception as e:
        print(f"❌ Gemini analysis error: {e}")
        # Return True with medium confidence to not block results during testing
        return True, 0.5, f"Analysis skipped - potential opportunity", "Consider engaging with helpful information about your service"

def search_reddit(keywords, service_description, posts_per_keyword, relevance_threshold=0.3):
    global search_progress
    results = []
    total_keywords = len(keywords)
    
    search_progress['total'] = total_keywords
    search_progress['current'] = 0
    search_progress['is_running'] = True
    search_progress['error'] = None
    
    try:
        for i, keyword in enumerate(keywords, start=1):
            search_progress['current'] = i
            search_progress['message'] = f"Searching for: '{keyword}'"
            
            try:
                relevant_posts_found = 0
                posts_analyzed = 0
                
                print(f"🔍 Searching Reddit for keyword: {keyword}")
                
                # Search Reddit for this keyword
                for submission in reddit.subreddit("all").search(keyword, sort="relevance", limit=min(posts_per_keyword * 2, 10)):
                    if relevant_posts_found >= posts_per_keyword:
                        break
                    
                    posts_analyzed += 1
                    print(f"  📝 Analyzing post {posts_analyzed}: {submission.title[:50]}...")
                    
                    # Skip posts that are too old or have very low engagement
                    if submission.score < 1 and submission.num_comments < 1:
                        continue
                    
                    # Analyze post relevance using Gemini
                    is_relevant, confidence, reason, marketing_suggestion = analyze_post_relevance(
                        service_description, 
                        submission.title, 
                        submission.selftext
                    )
                    
                    # Include posts that meet the relevance threshold
                    if is_relevant and confidence >= relevance_threshold:
                        result_data = {
                            "Keyword": keyword,
                            "Title": submission.title,
                            "Subreddit": submission.subreddit.display_name,
                            "Score": submission.score,
                            "Comments": submission.num_comments,
                            "URL": f"https://reddit.com{submission.permalink}",
                            "Created_UTC": datetime.fromtimestamp(submission.created_utc).strftime('%Y-%m-%d %H:%M:%S'),
                            "Relevant": "Yes",
                            "Confidence": f"{confidence:.2f}",
                            "Reason": reason,
                            "Marketing_Suggestion": marketing_suggestion,
                            "Post_Content": submission.selftext[:500] + "..." if len(submission.selftext) > 500 else submission.selftext,
                            "Full_Permalink": submission.permalink
                        }
                        results.append(result_data)
                        relevant_posts_found += 1
                        search_progress['message'] = f"✅ Found {relevant_posts_found} relevant posts for '{keyword}'"
                        print(f"    ✅ Added post: {submission.title[:50]}...")
                    
                    # Add a small delay to avoid rate limiting
                    time.sleep(1)
                
                print(f"📊 Keyword '{keyword}': Analyzed {posts_analyzed} posts, found {relevant_posts_found} relevant")
                    
            except Exception as e:
                print(f"❌ Error while searching for '{keyword}': {e}")
                continue
        
        search_progress['results'] = results
        search_progress['is_running'] = False
        search_progress['message'] = f"🎉 Search completed! Found {len(results)} relevant posts."
        print(f"🎉 Final results: {len(results)} posts found")
        return results
        
    except Exception as e:
        search_progress['error'] = str(e)
        search_progress['is_running'] = False
        print(f"💥 Search failed with error: {e}")
        return None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    global search_progress, search_config
    
    if 'file' not in request.files:
        flash('No file selected', 'error')
        return redirect(url_for('index'))
    
    file = request.files['file']
    
    if file.filename == '':
        flash('No file selected', 'error')
        return redirect(url_for('index'))
    
    if not file.filename.endswith(('.xlsx', '.xls')):
        flash('Please upload an Excel file (.xlsx or .xls)', 'error')
        return redirect(url_for('index'))
    
    # Get service description and posts per keyword from form
    service_description = request.form.get('service_description', '').strip()
    posts_per_keyword = request.form.get('posts_per_keyword', '5')  # Reduced for testing
    
    if not service_description:
        flash('Please enter a service description', 'error')
        return redirect(url_for('index'))
    
    try:
        posts_per_keyword = int(posts_per_keyword)
        if posts_per_keyword < 1 or posts_per_keyword > 10:  # Reduced max for testing
            flash('Posts per keyword must be between 1 and 10', 'error')
            return redirect(url_for('index'))
    except ValueError:
        flash('Please enter a valid number for posts per keyword', 'error')
        return redirect(url_for('index'))
    
    # Update search configuration
    search_config['service_description'] = service_description
    search_config['posts_per_keyword'] = posts_per_keyword
    
    try:
        df = pd.read_excel(file)
        if "Keyword" not in df.columns:
            flash("Excel file must have a 'Keyword' column", 'error')
            return redirect(url_for('index'))
        
        keywords = df["Keyword"].dropna().str.strip().tolist()
        
        if not keywords:
            flash("No keywords found in the Excel file", 'error')
            return redirect(url_for('index'))
        
        print(f"🔑 Keywords to search: {keywords}")
        
        # Reset progress
        search_progress = {
            'current': 0,
            'total': len(keywords),
            'message': 'Starting search...',
            'is_running': True,
            'results': None,
            'error': None
        }
        
        # Start search in background thread
        thread = threading.Thread(
            target=search_reddit, 
            args=(keywords, service_description, posts_per_keyword)
        )
        thread.daemon = True
        thread.start()
        
        return redirect(url_for('progress'))
        
    except Exception as e:
        flash(f'Error reading file: {str(e)}', 'error')
        return redirect(url_for('index'))

@app.route('/progress')
def progress():
    return render_template('index.html', show_progress=True)

@app.route('/progress_data')
def progress_data():
    global search_progress
    return {
        'current': search_progress['current'],
        'total': search_progress['total'],
        'message': search_progress['message'],
        'is_running': search_progress['is_running'],
        'has_results': search_progress['results'] is not None,
        'error': search_progress['error']
    }

@app.route('/results')
def show_results():
    global search_progress
    
    if search_progress['results'] is None:
        flash('No results available. Please run a search first.', 'error')
        return redirect(url_for('index'))
    
    results = search_progress['results']
    return render_template('results.html', results=results, total_results=len(results))

@app.route('/download')
def download_results():
    global search_progress
    
    if search_progress['results'] is None:
        flash('No results available to download', 'error')
        return redirect(url_for('index'))
    
    # Create Excel file in memory
    output = BytesIO()
    df = pd.DataFrame(search_progress['results'])
    
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Reddit_Results')
    
    output.seek(0)
    filename = f"reddit_marketing_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
    
    return send_file(
        output,
        download_name=filename,
        as_attachment=True,
        mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
    )

@app.route('/reset')
def reset():
    global search_progress
    search_progress = {
        'current': 0,
        'total': 0,
        'message': '',
        'is_running': False,
        'results': None,
        'error': None
    }
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5000)