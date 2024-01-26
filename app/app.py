import streamlit as st


def analyze_sentiment(text):
    return "Positive" 

def main():
    st.title("Text Sentiment Analysis")

    # Input text area
    user_input = st.text_area("Enter your text here:")

    # Analyze button
    if st.button("Analyze Sentiment"):
        if user_input:
            result = analyze_sentiment(user_input)
            st.success(f"Sentiment: {result}")
        else:
            st.warning("Please enter some text before analyzing.")


if __name__ == "__main__":
    main()
