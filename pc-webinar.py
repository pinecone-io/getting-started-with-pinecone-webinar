#!/usr/bin/env python3
"""CLI tool for Pinecone webinar commands."""

# Load environment variables before any other imports
from dotenv import load_dotenv
load_dotenv()

import argparse
import sys

from assistant import load as assistant_load
from assistant import prompt as assistant_prompt
from database import load as database_load
from database import query as database_query
# Lazy import for data_download to avoid kaggle authentication on import


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description='Pinecone webinar CLI tool',
        prog='pc-webinar'
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Data download command
    data_download_parser = subparsers.add_parser(
        'data-download',
        help='Download the Steam dataset from Kaggle'
    )
    
    # Database commands
    db_load_parser = subparsers.add_parser(
        'database-load',
        help='Load data into the database'
    )
    
    db_query_parser = subparsers.add_parser(
        'database-query',
        help='Query the database'
    )
    db_query_parser.add_argument(
        'query',
        help='Search query string'
    )
    db_query_parser.add_argument(
        '-m', '--mode',
        choices=['hybrid', 'semantic', 'lexical'],
        default='hybrid',
        help='Search mode: hybrid (default), semantic (dense only), or lexical (sparse only)'
    )
    db_query_parser.add_argument(
        '-t', '--top-k',
        type=int,
        default=10,
        help='Number of results to return (default: 10)'
    )
    
    # Assistant commands
    asst_load_parser = subparsers.add_parser(
        'assistant-load',
        help='Load data for the assistant'
    )
    
    asst_prompt_parser = subparsers.add_parser(
        'assistant-prompt',
        help='Prompt the assistant'
    )
    asst_prompt_parser.add_argument(
        'prompt',
        help='Prompt/question to ask the assistant'
    )
    asst_prompt_parser.add_argument(
        '-m', '--model',
        choices=['gpt-4o', 'gpt-4.1', 'gpt-5', 'o4-mini', 'claude-3-5-sonnet', 'claude-3-7-sonnet', 'gemini-2.5-pro'],
        default='gpt-4o',
        help='Model to use for the assistant (default: gpt-4o)'
    )
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    # Route to appropriate command handler
    if args.command == 'data-download':
        # Lazy import to avoid kaggle authentication when not needed
        from data import download as data_download
        exit_code = data_download.main()
        sys.exit(exit_code)
    elif args.command == 'database-load':
        database_load.main()
    elif args.command == 'database-query':
        database_query.main(args.query, args.mode, args.top_k)
    elif args.command == 'assistant-load':
        assistant_load.main()
    elif args.command == 'assistant-prompt':
        assistant_prompt.main(args.prompt, args.model)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()

