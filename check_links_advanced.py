#!/usr/bin/env python3
import re
import os
import glob
from collections import defaultdict

def check_link_relevance(url, text, file_path):
    """Check if a link is relevant to its context."""
    issues = []
    
    # Common mismatches
    if 'face' in text.lower() and 'face' not in url.lower():
        if not any(term in url.lower() for term in ['biometric', 'recognition', 'detection', 'cv', 'vision']):
            issues.append('Face-related text but URL seems unrelated')
    
    if 'speech' in text.lower() and not any(term in url.lower() for term in ['speech', 'audio', 'voice', 'asr', 'tts']):
        issues.append('Speech-related text but URL seems unrelated')
    
    if 'llm' in text.lower() and not any(term in url.lower() for term in ['llm', 'language', 'gpt', 'bert', 'transformer', 'nlp']):
        issues.append('LLM-related text but URL seems unrelated')
    
    # Check for dataset links
    if any(term in text.lower() for term in ['dataset', 'data set', 'benchmark']):
        if not any(term in url.lower() for term in ['dataset', 'data', 'benchmark', 'corpus', 'collection']):
            issues.append('Dataset mentioned but URL seems unrelated')
    
    # Check for paper links
    if any(term in text.lower() for term in ['paper', 'arxiv', 'research']):
        if not any(term in url.lower() for term in ['arxiv', 'paper', 'pdf', 'acm', 'ieee', 'springer']):
            issues.append('Paper/research mentioned but URL seems unrelated')
    
    return issues

def extract_urls_with_context(file_path):
    """Extract URLs with surrounding context."""
    urls = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
        for i, line in enumerate(lines, 1):
            # Pattern 1: [text](url)
            markdown_links = re.findall(r'\[([^\]]+)\]\(([^)]+)\)', line)
            for text, url in markdown_links:
                if url and not url.startswith('#'):
                    # Get context from surrounding lines
                    context_start = max(0, i - 3)
                    context_end = min(len(lines), i + 2)
                    context = ''.join(lines[context_start:context_end])
                    
                    urls.append({
                        'file': file_path,
                        'text': text,
                        'url': url,
                        'line': i,
                        'context': context.strip(),
                        'type': 'markdown'
                    })
    
    return urls

def check_common_broken_patterns(url):
    """Check for common patterns that indicate broken or problematic links."""
    issues = []
    
    # Check for common broken GitHub patterns
    if 'github.com' in url:
        # Raw GitHub links that should use githubusercontent
        if '/blob/' in url and any(ext in url for ext in ['.png', '.jpg', '.gif', '.svg']):
            issues.append('GitHub blob link to image - should use raw.githubusercontent.com')
        
        # Check for deleted repos (common patterns)
        if any(pattern in url for pattern in ['/archive/', '/releases/download/']):
            if url.endswith('/'):
                issues.append('Malformed GitHub release/archive URL')
    
    # Check for outdated documentation patterns
    if any(domain in url for domain in ['readthedocs.io', 'docs.']):
        if any(pattern in url for pattern in ['/en/latest/', '/stable/', '/v1.', '/v2.']):
            issues.append('Version-specific documentation link - may be outdated')
    
    # Check for temporary/local URLs
    if any(pattern in url for pattern in ['localhost', '127.0.0.1', '192.168.', '10.0.']):
        issues.append('Local/internal URL')
    
    # Check for URL shorteners (often break)
    if any(domain in url for domain in ['bit.ly', 'tinyurl.com', 'goo.gl', 't.co']):
        issues.append('URL shortener - may be broken')
    
    return issues

# Main execution
base_dir = '/home/umit/CLAUDE_PROJECT/Awesome-AI-Resources'
all_urls = []

# Find all markdown files
md_files = glob.glob(os.path.join(base_dir, '**/*.md'), recursive=True)

print(f"Found {len(md_files)} markdown files")
print("Extracting URLs with context...")

for md_file in md_files:
    urls = extract_urls_with_context(md_file)
    all_urls.extend(urls)

print(f"\nTotal URLs found: {len(all_urls)}")

# Check for issues
all_issues = defaultdict(list)

for url_info in all_urls:
    url = url_info['url']
    text = url_info['text']
    
    # Check relevance
    relevance_issues = check_link_relevance(url, text, url_info['file'])
    if relevance_issues:
        for issue in relevance_issues:
            all_issues['irrelevant_links'].append({
                **url_info,
                'issue': issue
            })
    
    # Check common broken patterns
    broken_patterns = check_common_broken_patterns(url)
    if broken_patterns:
        for issue in broken_patterns:
            all_issues['broken_patterns'].append({
                **url_info,
                'issue': issue
            })
    
    # Check for duplicate/similar URLs
    # This is simplified - in reality would need more sophisticated deduplication
    
    # Check for missing HTTPS
    if url.startswith('http://') and not any(pattern in url for pattern in ['localhost', '127.0.0.1']):
        all_issues['http_not_https'].append(url_info)

# Report detailed issues
print("\n=== DETAILED LINK ISSUES REPORT ===\n")

if all_issues['irrelevant_links']:
    print(f"## Potentially Irrelevant Links ({len(all_issues['irrelevant_links'])} found):")
    for item in all_issues['irrelevant_links'][:20]:
        rel_path = os.path.relpath(item['file'], base_dir)
        print(f"\n  File: {rel_path}:{item['line']}")
        print(f"  Text: '{item['text']}'")
        print(f"  URL: {item['url']}")
        print(f"  Issue: {item['issue']}")
    if len(all_issues['irrelevant_links']) > 20:
        print(f"\n  ... and {len(all_issues['irrelevant_links']) - 20} more")

if all_issues['broken_patterns']:
    print(f"\n## Links with Broken Patterns ({len(all_issues['broken_patterns'])} found):")
    for item in all_issues['broken_patterns'][:20]:
        rel_path = os.path.relpath(item['file'], base_dir)
        print(f"\n  File: {rel_path}:{item['line']}")
        print(f"  URL: {item['url']}")
        print(f"  Issue: {item['issue']}")
    if len(all_issues['broken_patterns']) > 20:
        print(f"\n  ... and {len(all_issues['broken_patterns']) - 20} more")

# Find specific categories of issues
print("\n## Specific Issue Categories:\n")

# Check for Google Colab links
colab_links = [u for u in all_urls if 'colab.research.google.com' in u['url']]
if colab_links:
    print(f"### Google Colab Links ({len(colab_links)} found):")
    for item in colab_links[:5]:
        rel_path = os.path.relpath(item['file'], base_dir)
        print(f"  - {rel_path}:{item['line']} - {item['url']}")

# Check for arXiv links
arxiv_links = [u for u in all_urls if 'arxiv.org' in u['url']]
print(f"\n### ArXiv Links ({len(arxiv_links)} found) - Sample check for validity")

# Check for Kaggle links  
kaggle_links = [u for u in all_urls if 'kaggle.com' in u['url']]
print(f"\n### Kaggle Links ({len(kaggle_links)} found) - May require login")

# Summary
print(f"\n=== FINAL SUMMARY ===")
print(f"Total URLs analyzed: {len(all_urls)}")
print(f"Potentially irrelevant links: {len(all_issues['irrelevant_links'])}")
print(f"Links with broken patterns: {len(all_issues['broken_patterns'])}")
print(f"HTTP links (should be HTTPS): {len(all_issues['http_not_https'])}")