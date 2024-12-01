import base64
import cv2
from anthropic import Anthropic
from tavily import TavilyClient
import os
from dotenv import load_dotenv
import aiohttp
from bs4 import BeautifulSoup
import asyncio
import PyPDF2
import io
import urllib.parse

# Only load .env if it exists
if os.path.exists(".env"):
    load_dotenv()

async def scrape_url(url):
    async with aiohttp.ClientSession() as session:
        try:
            # Check if URL is a PDF
            is_pdf = url.lower().endswith('.pdf') or '.pdf' in urllib.parse.urlparse(url).path.lower()
            
            async with session.get(url) as response:
                if response.status == 200:
                    if is_pdf:
                        # Handle PDF content
                        pdf_content = await response.read()
                        pdf_text = extract_pdf_text(pdf_content)
                        return pdf_text
                    else:
                        # Handle HTML content
                        html = await response.text()
                        soup = BeautifulSoup(html, 'html.parser')
                        # Remove script and style elements
                        for script in soup(["script", "style"]):
                            script.decompose()
                        return soup.get_text()
                return ""
        except Exception as e:
            print(f"Error scraping URL {url}: {str(e)}")
            return ""

def extract_pdf_text(pdf_content):
    try:
        # Create a PDF file-like object from the content
        pdf_file = io.BytesIO(pdf_content)
        
        # Create PDF reader object
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        
        # Extract text from all pages
        text = []
        for page in pdf_reader.pages:
            text.append(page.extract_text())
        
        # Join all pages with newlines
        return "\n".join(text)
    except Exception as e:
        print(f"Error extracting PDF text: {str(e)}")
        return ""

class AIAnalyzer:
    def __init__(self):
        api_key = os.environ.get('ANTHROPIC_API_KEY')
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable is not set")
            
        # Initialize without proxies
        self.anthropic = Anthropic(
            api_key=api_key,
        )
        
        tavily_key = os.environ.get('TAVILY_API_KEY')
        if not tavily_key:
            raise ValueError("TAVILY_API_KEY environment variable is not set")
            
        self.tavily_client = TavilyClient(api_key=tavily_key)
        
        self.last_tavily_results = []
        
    def get_image_base64(self, image):
        _, buffer = cv2.imencode('.jpg', image)
        return base64.b64encode(buffer).decode('utf-8')
        
    async def analyze_snapshot_claude(self, image, ocr_texts=None, barcode_texts=None):
        try:
            image_base64 = self.get_image_base64(image)
            
            detected_text = []
            if ocr_texts:
                detected_text.extend(ocr_texts)
            if barcode_texts:
                detected_text.extend(barcode_texts)
            
            text_content = "\n".join(detected_text) if detected_text else "No text detected"
            
            # Get Claude analysis
            response = self.anthropic.messages.create(
                model="claude-3-haiku-20240307",
                messages=[{
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": image_base64
                            }
                        },
                        {
                            "type": "text",
                            "text": f"Analyse this image and the following detected text. Use it if relevant, otherwise ignore it:\n{text_content}"
                        }
                    ]
                }],
                system="You are an expert computer vision and OCR analyst. Your task is to:\n1. Analyse the image and any ocr text and/or barcodes/qr codes detected\n2. Extract any visible text, numbers, codes, or identifiable information\n3. If no text is found, describe what is visible in the image that might be relevant\n4. Format your response exactly as follows:\n\nAnalysis:\n[Your detailed analysis here]\n\nSearch Terms:\n[List 2-3 specific search terms, separated by commas]",
                max_tokens=1024
            )

            if not response or not response.content:
                return "Error: No response from Claude"
            
            claude_analysis = response.content[0].text
            
            # Store search terms for later Tavily use
            if "Search Terms:" in claude_analysis:
                terms_section = claude_analysis.split("Search Terms:")[-1].strip()
                if terms_section.lower() != "none":
                    self.last_search_terms = [term.strip() for term in terms_section.split(",") if term.strip()][:3]
            
            return claude_analysis
            
        except Exception as e:
            return f"Error during Claude analysis: {str(e)}"

    async def analyze_snapshot_tavily(self):
        try:
            if not hasattr(self, 'last_search_terms') or not self.last_search_terms:
                return []
            
            tavily_results = []
            for term in self.last_search_terms:
                try:
                    search_result = self.tavily_client.search(
                        query=term,
                        include_answer=True,
                        search_depth="advanced",
                        max_results=1
                    )
                    if search_result and search_result.get('results'):
                        result_entry = {
                            'term': term,
                            'title': search_result['results'][0].get('title', ''),
                            'content': search_result['results'][0].get('content', '')[:200] + '...',
                            'url': search_result['results'][0].get('url', '')
                        }
                        tavily_results.append(result_entry)
                except Exception as search_error:
                    print(f"Tavily search error for term '{term}': {search_error}")
            
            # Store results for context analysis
            if tavily_results:
                self.last_tavily_results = tavily_results
            
            return tavily_results
            
        except Exception as e:
            print(f"Error during Tavily analysis: {str(e)}")
            return []
        
    async def analyze_context(self, question):
        try:
            if not self.last_tavily_results:
                return "Please perform an AI snapshot search first to get context for analysis"
            
            urls = [result['url'] for result in self.last_tavily_results]
            if not urls:
                return "No URLs available for context analysis"
            
            # Scrape content from URLs
            contents = await asyncio.gather(*[scrape_url(url) for url in urls])
            
            # Process the combined context: remove empty lines and limit characters
            processed_contents = []
            for content in contents:
                # Remove empty lines and excessive whitespace
                cleaned_content = '\n'.join(line.strip() for line in content.splitlines() if line.strip())
                processed_contents.append(cleaned_content)
            
            combined_context = ' '.join(processed_contents)
            
            # Limit to 850000 characters
            if len(combined_context) > 850000:
                combined_context = combined_context[:850000]
            
            # Create the prompt with question first
            prompt = f"Question: {question}\n\nPlease answer the above question using the following context:\n\n{combined_context}"
            
            # Log the complete prompt to console
            print("\n=== CLAUDE CONTEXT ANALYSIS PROMPT ===")
            print("System prompt:", "You are an expert analyst. Your task is to answer the user's question based on the provided context. Focus on relevant information and provide a clear, concise summary.")
            print("\nUser prompt:", prompt)
            print("\nContext length:", len(combined_context), "characters")
            print("=====================================\n")
            
            # Get Claude analysis
            response = self.anthropic.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=1024,
                messages=[{
                    "role": "user",
                    "content": prompt
                }],
                system="You are an expert analyst. Your task is to answer the user's question based on the provided context. Focus on relevant information and provide a clear, concise summary.",
            )
            
            return response.content[0].text if response and response.content else "No analysis available"
            
        except Exception as e:
            return f"Error during context analysis: {str(e)}"