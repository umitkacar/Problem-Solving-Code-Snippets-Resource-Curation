#!/usr/bin/env python3
import re
import os
import glob
from collections import defaultdict

def extract_urls_from_markdown(file_path):
    """Extract all URLs from a markdown file."""
    urls = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
        
        # Pattern 1: [text](url)
        markdown_links = re.findall(r'\[([^\]]+)\]\(([^)]+)\)', content)
        for text, url in markdown_links:
            if url and not url.startswith('#'):  # Skip anchor links
                urls.append({
                    'file': file_path,
                    'text': text,
                    'url': url,
                    'line': None,
                    'type': 'markdown'
                })
        
        # Pattern 2: Direct URLs (http/https)
        direct_urls = re.findall(r'(?<![(\[])https?://[^\s\)\]]+', content)
        for url in direct_urls:
            urls.append({
                'file': file_path,
                'text': None,
                'url': url,
                'line': None,
                'type': 'direct'
            })
        
        # Pattern 3: Reference-style links [text][ref]
        ref_links = re.findall(r'\[([^\]]+)\]\[([^\]]+)\]', content)
        ref_definitions = re.findall(r'^\[([^\]]+)\]:\s*(.+)$', content, re.MULTILINE)
        ref_dict = dict(ref_definitions)
        
        for text, ref_id in ref_links:
            if ref_id in ref_dict:
                urls.append({
                    'file': file_path,
                    'text': text,
                    'url': ref_dict[ref_id],
                    'line': None,
                    'type': 'reference'
                })
    
    # Add line numbers
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for url_info in urls:
            for i, line in enumerate(lines, 1):
                if url_info['url'] in line:
                    url_info['line'] = i
                    break
    
    return urls

def categorize_issues(urls):
    """Categorize potential issues with URLs."""
    issues = defaultdict(list)
    
    for url_info in urls:
        url = url_info['url']
        
        # Check for HTTP (should be HTTPS)
        if url.startswith('http://') and not url.startswith('http://localhost'):
            issues['http_urls'].append(url_info)
        
        # Check for common broken link patterns
        if any(pattern in url for pattern in ['404', 'not-found', 'error']):
            issues['potential_404'].append(url_info)
        
        # Check for placeholder URLs
        if any(pattern in url for pattern in ['example.com', 'placeholder', 'your-', 'TODO']):
            issues['placeholder_urls'].append(url_info)
        
        # Check for local file references
        if url.startswith('file://') or (url.startswith('./') and not url.endswith('.md')):
            issues['local_references'].append(url_info)
        
        # Check for malformed URLs
        if ' ' in url or '\t' in url or '\n' in url:
            issues['malformed_urls'].append(url_info)
        
        # Check for LICENSE file reference (known missing)
        if 'LICENSE' in url and not url.endswith('.md'):
            issues['missing_license'].append(url_info)
    
    return issues

# Main execution
base_dir = '/home/umit/CLAUDE_PROJECT/Awesome-AI-Resources'
all_urls = []

# Find all markdown files
md_files = glob.glob(os.path.join(base_dir, '**/*.md'), recursive=True)

print(f"Found {len(md_files)} markdown files")
print("Extracting URLs...")

for md_file in md_files:
    urls = extract_urls_from_markdown(md_file)
    all_urls.extend(urls)

print(f"\nTotal URLs found: {len(all_urls)}")

# Categorize issues
issues = categorize_issues(all_urls)

# Report issues
print("\n=== URL ISSUES REPORT ===\n")

if issues['http_urls']:
    print(f"## HTTP URLs that should be HTTPS ({len(issues['http_urls'])} found):")
    for url_info in issues['http_urls'][:10]:  # Show first 10
        rel_path = os.path.relpath(url_info['file'], base_dir)
        print(f"  - {rel_path}:{url_info['line']} - {url_info['url']}")
    if len(issues['http_urls']) > 10:
        print(f"  ... and {len(issues['http_urls']) - 10} more")

if issues['potential_404']:
    print(f"\n## Potential 404/broken URLs ({len(issues['potential_404'])} found):")
    for url_info in issues['potential_404']:
        rel_path = os.path.relpath(url_info['file'], base_dir)
        print(f"  - {rel_path}:{url_info['line']} - {url_info['url']}")

if issues['placeholder_urls']:
    print(f"\n## Placeholder URLs ({len(issues['placeholder_urls'])} found):")
    for url_info in issues['placeholder_urls']:
        rel_path = os.path.relpath(url_info['file'], base_dir)
        print(f"  - {rel_path}:{url_info['line']} - {url_info['url']}")

if issues['local_references']:
    print(f"\n## Local file references ({len(issues['local_references'])} found):")
    for url_info in issues['local_references']:
        rel_path = os.path.relpath(url_info['file'], base_dir)
        print(f"  - {rel_path}:{url_info['line']} - {url_info['url']}")

if issues['malformed_urls']:
    print(f"\n## Malformed URLs ({len(issues['malformed_urls'])} found):")
    for url_info in issues['malformed_urls']:
        rel_path = os.path.relpath(url_info['file'], base_dir)
        print(f"  - {rel_path}:{url_info['line']} - {url_info['url']}")

if issues['missing_license']:
    print(f"\n## Missing LICENSE file references ({len(issues['missing_license'])} found):")
    for url_info in issues['missing_license']:
        rel_path = os.path.relpath(url_info['file'], base_dir)
        print(f"  - {rel_path}:{url_info['line']} - {url_info['url']}")

# Summary
print(f"\n=== SUMMARY ===")
print(f"Total URLs checked: {len(all_urls)}")
print(f"Total issues found: {sum(len(v) for v in issues.values())}")