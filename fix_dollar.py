import os

def fix_dollar_escape(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        content = f.read()

    # The issue: $0.01/1M tokens)  -> should be \$0.01/1M tokens), otherwise $ opens math mode
    content = content.replace("约 $0.01/1M", "约 \\$0.01/1M")

    # In en 
    content = content.replace("approx $0.01/1M", "approx \\$0.01/1M")
    
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(content)

fix_dollar_escape("investor_deck_mt_lnn_zh.tex")
fix_dollar_escape("investor_deck_mt_lnn.tex")
