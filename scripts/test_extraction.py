#!/usr/bin/env python3
"""Test script for data extraction modules."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def test_whatsapp():
    """Test WhatsApp message extraction."""
    print("Testing WhatsApp extraction...")
    
    from brainlayer.pipeline.extract_whatsapp import extract_whatsapp_messages, analyze_writing_style
    
    try:
        messages = list(extract_whatsapp_messages(limit=100, only_from_me=True))
        print(f"✓ Extracted {len(messages)} messages")
        
        if messages:
            print(f"\nSample message:")
            print(f"  Text: {messages[0]['text'][:100]}...")
            print(f"  Timestamp: {messages[0]['datetime']}")
            print(f"  From me: {messages[0]['is_from_me']}")
            
            # Analyze style
            style = analyze_writing_style(messages)
            print(f"\nWriting style analysis:")
            print(f"  Total messages: {style['total_messages']}")
            print(f"  Avg length: {style['avg_message_length']:.0f} chars")
            print(f"  Emoji rate: {style['emoji_usage_rate']:.2f} per message")
        
        return True
        
    except FileNotFoundError as e:
        print(f"✗ WhatsApp database not found: {e}")
        return False
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_claude_desktop():
    """Test Claude desktop chat extraction."""
    print("\nTesting Claude desktop extraction...")
    
    from brainlayer.pipeline.extract_claude_desktop import get_claude_indexeddb_path
    
    db_path = get_claude_indexeddb_path()
    
    if db_path.exists():
        print(f"✓ Found Claude IndexedDB at: {db_path}")
        print("  Note: Extraction requires manual export or plyvel library")
        return True
    else:
        print(f"✗ Claude IndexedDB not found at: {db_path}")
        return False


def test_communication_analyzer():
    """Test communication pattern analyzer."""
    print("\nTesting communication analyzer...")
    
    from brainlayer.pipeline.analyze_communication import CommunicationAnalyzer
    
    try:
        analyzer = CommunicationAnalyzer()
        
        # Add sample data
        analyzer.user_messages.append({
            'text': 'Hey! How are you doing?',
            'source': 'test',
            'timestamp': None
        })
        analyzer.user_messages.append({
            'text': 'Can you help me with something?',
            'source': 'test',
            'timestamp': None
        })
        
        # Analyze
        style = analyzer.analyze_writing_style()
        print(f"✓ Analyzer working")
        print(f"  Messages analyzed: {style['total_messages_analyzed']}")
        print(f"  Avg length: {style['avg_message_length']:.0f} chars")
        
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("="*70)
    print("BrainLayer Data Extraction Tests")
    print("="*70 + "\n")
    
    results = {
        'WhatsApp': test_whatsapp(),
        'Claude Desktop': test_claude_desktop(),
        'Communication Analyzer': test_communication_analyzer(),
    }
    
    print("\n" + "="*70)
    print("Results:")
    print("="*70)
    
    for name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {name}")
    
    all_passed = all(results.values())
    print("\n" + ("All tests passed!" if all_passed else "Some tests failed"))
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
