import os
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_openai_connection():
    try:
        # Initialize the OpenAI client
        client = OpenAI()
        
        # Make a simple completion request
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Say 'Hello! Your OpenAI API key is working correctly!'"}
            ],
            max_tokens=50
        )
        print(response)
        # Print the response
        print("\nAPI Key Test Results:")
        print("--------------------") 
        print("Status: Success!")
        print("Response:", response.choices[0].message.content)
        print("\nYour OpenAI API key is working correctly!")
        
    except Exception as e:
        print("\nAPI Key Test Results:")
        print("--------------------")
        print("Status: Failed!")
        print("Error:", str(e))
        print("\nPlease check your OpenAI API key and try again.")

if __name__ == "__main__":
    test_openai_connection()
