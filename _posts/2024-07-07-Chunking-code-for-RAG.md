---
layout: post
title: "Chunking code for RAG; parsing-recursion-stack"
date: "2024-07-07"
mathjax: true
status: [Code samples]
categories: [Code]
---


#### **Main Idea**

For RAG on code, a natural way to chunk code blocks is by parsing the code using the grammar of the programming language we are working with. This ensures the code is chunked according to semantic blocks, instead of arbitrary max-char threshold cut-offs. 

#### **Method**

The `tree-sitter` library allows us to produce concrete syntax trees for various programming language. In this example, we're working with parsing C#. 

#### <u>Build the relevant grammar and parser</u>

{% highlight python %}
import os
import subprocess
from tree_sitter import Language, Parser, Node

def build_parser(prog_lang:str=""):
 
  if not os.path.exists(f'cache/build/{prog_lang}.so'):
    cmd = f"git clone https://github.com/tree-sitter/tree-sitter-{prog_lang}"
    store = f"cache/tree-sitter-{prog_lang}"
    cmd = cmd + " " + store
    subprocess.run(cmd, shell=True)
  
  Language.build_library(f'cache/build/{prog_lang}.so', prog_lang)
  language_ = Language(f'cache/build/{prog_lang}.so', prog_lang)

  parser = Parser()
  parser.set_language(language_)

{% endhighlight %}

<br>

#### <u>Read in code repository and parse each file. </u>

The output of the tree-sitter parser is a concrete syntax tree object, with nodes representing different building blocks
of the code. Each node can have child nodes that represent sub-building blocks. Since the
parser uses a programming language specific grammar, the nodes represent expressions,
statements, classes, functions and other constructs. 

{% highlight python %}

from tqdm import tqdm
from pathlib import Path
code_files = list(Path(code_folder).rglob("*.cs"))

all_file_chunks = []

# this code is simplified to just show the parsing and chunking. In actual implementation, we would store more metadata along with the chunks.

for filepath in tqdm(code_files):
  text = filepath.read_text()
  tree = parser.parse(bytes(text, 'utf-8'))
  new_chunks = chunker.chunk_node(tree.root_node, current_chunk="")
  all_file_chunks.append(new_chunks)


{% endhighlight %}

<br>

#### <u> Breakdown tree recursively using a stack </u>

We would like to have a min and max size of each code block (chunk) so that we can retrieve
semantically meaningful blocks of code. The chunking logic core functionality is implemented in
`chunk_node`, with a helper method `_chunk_node`. 

The logic is to *recursively* traverse the syntax tree, breaking it down into chunks that
respect the size constraints. The first `if-condition` is straightforward, if the node is too
large, apply the function recursively. 

For the main `else-condition`, we need to check if the current chunk can be collected as
a valid chunk, or if it needs to be combined with the `new_text`. We make use of a stack data structure so that we are always working with either adding, or combining the most recent chunk at the top of the stack(LIFO). Once this chunk has valid size constraints, we add it to our `new_chunks` and pop it off the top of the stack.


{% highlight python %}
from collections import deque
from tree_sitter import Node

class CodeChunker():
  def __init__(self, MAX_CHARS: int=1500, MIN_CHARS: int=200):
    self.MIN_CHARS = MIN_CHARS
    self.MAX_CHARS = MAX_CHARS

  def chunk_node(self, node: Node) -> list[str]:
    new_chunks = []
    nodeStack = deque()
    self._chunk_node(node, nodeStack, new_chunks)
    return new_chunks

  def _chunk_node(self, node: Node, nodeStack: deque, new_chunks: list[str]):
    for child in node.children:
      if child.end_byte - child.start_byte > self.MAX_CHARS:
        self._check_minsize_add(new_chunks, nodeStack)
        self._chunk_node(node=child,
                         nodeStack=nodeStack,
                         new_chunks=new_chunks)
      else:
        new_text = child.text.decode('utf-8')
        
        if not nodeStack:
          nodeStack.append(new_text)
          continue  

        if len(nodeStack[-1]) >= self.MIN_CHARS:
          new_chunks.append(nodeStack.pop())
          nodeStack.append(new_text)
        else:
          nodeStack[-1] = " ".join([nodeStack[-1], new_text])


    # if there are no other child nodes to combine with, we just collect all the remaining chunks in the stack.
    while len(nodeStack !=0):
      new_chunks.append(nodeStack.pop())

  def _check_minsize_add(self, new_chunks, nodeStack):
      current_chunk = "" if not nodeStack else nodeStack[-1]
      if len(current_chunk) >= self.MIN_CHARS:
        new_chunks.append(nodeStack.pop())
{% endhighlight %}

<br>

#### **Implementation Notes**

* The [core library](https://tree-sitter.github.io/tree-sitter/) is written in C, but has a python wrapper for each language.

* The python wrapper for the library may not be on pip. If so, we need to install this directly from git. I was working with C# and used version 0.21.3 for backward compatibility with the tree-sitter C library. The wrapper is not completely intuitive, the error messages was not informative, and the argument for the language was inconsistent. e.g., `Language('cache/build/c-sharp.so', 'c_sharp')` 

* We can inspect the node types with node.type and further include this information in the meta-data of the chunks.

* I did not implement this with a stack originally, and did `" ".join(current_chunk, new_text)` passing `current_chunk` as an argument under the recursive call. This resulted in the same `current_chunk` was being joined to multiple `new_text`. The stack is necessary as a mutable object which is *pass-by-reference* to the recursive call, allowing the function to modify the original object. 

* All our operations involving the stack are O(1) as deque is a double linked list. This is an efficient way to implement our stack, as we only use the operations of append, pop, and inspecting the last element in the stack. If we
needed to access other elements, this probably wouldnt work because the inspection is O(n).

* Writing the unit test is straightforward; ensure that we can reconstruct the same code file, if all
  whitespaces are removed. e.g., `assert re.sub(r"\s+", "", text) == re.sub(r"\s+", "", "".join(chunks))`
