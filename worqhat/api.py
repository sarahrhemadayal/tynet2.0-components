import google.generativeai as genai


API_KEY = 'AIzaSyDdBLfAlky6LwI6mDy7ekghwIpgO0NADjQ'

genai.configure(api_key=API_KEY)
model = genai.GenerativeModel("gemini-1.5-flash")
response = model.generate_content("Explain how AI works")
print(response.text)
