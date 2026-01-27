#!/usr/bin/env python3
"""
Extract samples of non-english, emoji-heavy, and url-only content from Claude JSONL files
"""

import json
import glob
import re
import os
from typing import List, Dict, Any

def extract_samples():
    # Find JSONL files
    jsonl_files = glob.glob(os.path.expanduser('~/.claude/projects/**/*.jsonl'), recursive=True)
    
    non_english_samples = []
    emoji_samples = []
    url_only_samples = []
    
    # Patterns for detection
    emoji_pattern = re.compile(r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF\U00002600-\U000027BF\U0001F900-\U0001F9FF\U0001F700-\U0001F77F\U0001F780-\U0001F7FF\U0001F800-\U0001F8FF\U0001F100-\U0001F1FF\U0001F200-\U0001F2FF\U0001F000-\U0001F0FF]+')
    url_pattern = re.compile(r'^https?://[^\s]+$')
    
    # Hebrew, Arabic, Chinese, Japanese, Korean, Cyrillic patterns
    non_english_patterns = [
        re.compile(r'[\u0590-\u05FF]+'),  # Hebrew
        re.compile(r'[\u0600-\u06FF]+'),  # Arabic
        re.compile(r'[\u4e00-\u9fff]+'),  # Chinese
        re.compile(r'[\u3040-\u309f\u30a0-\u30ff]+'),  # Japanese
        re.compile(r'[\uac00-\ud7af]+'),  # Korean
        re.compile(r'[\u0400-\u04FF]+'),  # Cyrillic
    ]
    
    print(f"Scanning {len(jsonl_files)} JSONL files...")
    
    for file_path in jsonl_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f):
                    if line_num > 200:  # Limit lines per file
                        break
                    try:
                        data = json.loads(line.strip())
                        
                        # Extract content from various fields
                        content_fields = []
                        if 'content' in data:
                            content_fields.append(str(data['content']))
                        if 'message' in data and isinstance(data['message'], dict):
                            if 'content' in data['message']:
                                content_fields.append(str(data['message']['content']))
                        
                        for content in content_fields:
                            if not content or len(content.strip()) < 3:
                                continue
                                
                            # Check for non-English content
                            for pattern in non_english_patterns:
                                if pattern.search(content) and len(non_english_samples) < 15:
                                    non_english_samples.append({
                                        'file': os.path.basename(file_path),
                                        'line': line_num,
                                        'content': content[:300] + '...' if len(content) > 300 else content,
                                        'type': data.get('type', 'unknown'),
                                        'language_detected': 'hebrew' if re.search(r'[\u0590-\u05FF]+', content) else 'other'
                                    })
                                    break
                            
                            # Check for emoji-heavy content
                            emoji_matches = emoji_pattern.findall(content)
                            if len(emoji_matches) >= 2 and len(emoji_samples) < 15:
                                emoji_samples.append({
                                    'file': os.path.basename(file_path),
                                    'line': line_num,
                                    'content': content[:300] + '...' if len(content) > 300 else content,
                                    'type': data.get('type', 'unknown'),
                                    'emoji_count': len(emoji_matches)
                                })
                            
                            # Check for URL-only content
                            stripped_content = content.strip()
                            if url_pattern.match(stripped_content) and len(url_only_samples) < 15:
                                url_only_samples.append({
                                    'file': os.path.basename(file_path),
                                    'line': line_num,
                                    'content': content,
                                    'type': data.get('type', 'unknown')
                                })
                                
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            continue
    
    return non_english_samples, emoji_samples, url_only_samples

if __name__ == "__main__":
    non_english, emoji_heavy, url_only = extract_samples()
    
    print(f"\n=== NON-ENGLISH SAMPLES ({len(non_english)} found) ===")
    for i, sample in enumerate(non_english[:8]):
        print(f"\n{i+1}. Type: {sample['type']} | Language: {sample['language_detected']}")
        print(f"   File: {sample['file']}")
        print(f"   Content: {sample['content']}")
    
    print(f"\n=== EMOJI-HEAVY SAMPLES ({len(emoji_heavy)} found) ===")
    for i, sample in enumerate(emoji_heavy[:8]):
        print(f"\n{i+1}. Type: {sample['type']} | Emoji count: {sample['emoji_count']}")
        print(f"   File: {sample['file']}")
        print(f"   Content: {sample['content']}")
    
    print(f"\n=== URL-ONLY SAMPLES ({len(url_only)} found) ===")
    for i, sample in enumerate(url_only[:8]):
        print(f"\n{i+1}. Type: {sample['type']}")
        print(f"   File: {sample['file']}")
        print(f"   Content: {sample['content']}")
