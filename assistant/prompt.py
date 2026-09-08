"""Assistant prompt command implementation."""

from pinecone_plugins.assistant.models.chat import Message
from shared import utils


def display_response(response):
    """Display the assistant response and citations.
    
    Args:
        response: The chat response object from the assistant
    """
    # Display the response
    model_name = getattr(response, 'model', 'Unknown')
    print("\nAssistant Response:")
    print(f"Model: {model_name}")
    print("-" * 80)
    print(response.message.content)
    print("-" * 80)
    
    # Display citations if available
    if hasattr(response, 'citations') and response.citations:
        print("\nCitations:")
        for citation in response.citations:
            for ref in citation.references:
                file_info = ref.file
                print(f"  - {file_info.name}")
                if hasattr(ref, 'pages') and ref.pages:
                    print(f"    Pages: {', '.join(map(str, ref.pages))}")


def main(prompt: str, model: str = 'gpt-4o'):
    """Execute the assistant-prompt command.
    
    Args:
        prompt: The prompt/question to ask the assistant
        model: The model to use for the assistant (default: gpt-4o)
    """
    # Get the Pinecone assistant
    assistant = utils.get_assistant()
    
    # Create a message from the user prompt
    msg = Message(role="user", content=prompt)
    
    # Chat with the assistant using the specified model
    response = assistant.chat(messages=[msg], model=model)
    
    # Display the response and citations
    display_response(response)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Prompt the Pinecone Assistant')
    parser.add_argument('prompt', help='Prompt for the assistant')
    parser.add_argument(
        '-m', '--model',
        choices=['gpt-4o', 'gpt-4.1', 'gpt-5', 'o4-mini', 'claude-3-5-sonnet', 'claude-3-7-sonnet', 'gemini-2.5-pro'],
        default='gpt-4o',
        help='Model to use for the assistant (default: gpt-4o)'
    )
    
    args = parser.parse_args()
    main(args.prompt, args.model)

