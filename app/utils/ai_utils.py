import base64
import cv2
from anthropic import Anthropic
from tavily import TavilyClient
import os
from dotenv import load_dotenv

# Only load .env if it exists
if os.path.exists(".env"):
    load_dotenv()

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
        
    def get_image_base64(self, image):
        _, buffer = cv2.imencode('.jpg', image)
        return base64.b64encode(buffer).decode('utf-8')
        
    async def analyze_snapshot(self, image, ocr_texts=None, barcode_texts=None):
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
                return {"claude": "Error: No response from Claude", "tavily": []}

            claude_analysis = response.content[0].text

            # Extract search terms
            search_terms = []
            if "Search Terms:" in claude_analysis:
                terms_section = claude_analysis.split("Search Terms:")[-1].strip()
                if terms_section.lower() != "none":
                    search_terms = [term.strip() for term in terms_section.split(",") if term.strip()][:3]

            # Perform Tavily searches
            tavily_results = []
            if search_terms:
                for term in search_terms:
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

            return {
                "claude": claude_analysis,
                "tavily": tavily_results
            }
            
        except Exception as e:
            return {"claude": f"Error during analysis: {str(e)}", "tavily": []}