```@raw html
    <img src='assets/voronoimesh.png' alt='missing' class='center'><br>
```

````@eval
using Markdown
Markdown.parse("""
$(read("../../README.md",String))
""")
````