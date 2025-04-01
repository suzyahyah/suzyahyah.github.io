---
layout: post
title: "Data Extraction for Unstructured Document Data"
date: "2024-09-15"
mathjax: true
status: [Review]
shape: dot
categories: [NLP]
---

### **Preliminaries**

There are essentially these few elements that we want to extract from any set of documents.

| Data Type           | Description                                                                 |
|---------------------|-----------------------------------------------------------------------------|
| Doc-level Meta-data | doc-level information that should be semantically grouped together                     |
| Page-level Meta-data | page-level information that should be semantically grouped together                     |
| Raw text            | main text content                                                           |
| Images              | representing object or a scene in the real-world                             |
| Diagrams            | a (logic) process flow                                                              |
| Charts              | a summary of a data analysis result                                          |
| Tables              | tabular information, i.e. information in a grid                              |
| Captions            | are not considered main text content, as they are text descriptions that accompany images, diagrams, and tables |


<br>

The desired output from the data ingestion pipeline, is a “usable output” for ML, LLM, or RAG. All of these models operate on raw text, or raw image pixels, which benefit from maximum context of the data. 

That means, that the desired output of the data ingestion pipeline should include both the high-level meta-data of the document, and also the **low-level meta-data** of where content is found wrt to the document layout. 

This post describes only describes the data extraction pipeline, focusing on extraction of data elements.  **This post does not touch on handilng data processing at scale.**

<br>
### **1. Handling File Formats**

Documents hold semantic units of information, and can come in several typical file formats. We can briefly categorise this into Human-consumable formats (pptx, docx, PDFs, PNGs), and Machine consumable formats (html, xml, markdown, latex). 

#### **1.1 Machine Consumable Formats**
For machine-consumable formats, they follow known rules and are essentially encapsulated in tags, which allow their relatively smooth parsing by standard libraries. Before LMs, these were tedious to extract as it required a human programmer to go through the tags and write the code to extract them. Now with LLMs, it is almost straightforward to ask the LM to write code to extract the data.

*Note: Although it is possible to prompt LLM to convert the nested tag data to JSON, it is not advisable as it could be very expensive to keep running this operation.*


#### **1.2 Human Consumable Formats**
For human-consumable formats, the easiest to work with are docx (or PDF from docx), while the least flexible and most challenging formats are Pptx (or PDF from pptx). 

For the latter, we are trying to reverse engineer what was in a person’s mind when drawing on a blank canvas. This is highly problematic for three reasons; (1) decoding the Logical flow of information and (2) closeness in location does not always reflect closeness in meaning (3) unclear and non-standardised section delimiters. 

Contrast this with docx, which is typically in single column format.

#### **1.3 Standardising the File-format**

To avoid having to build a specialised pipeline for each of the file formats, the first step is to convert everything to PDF. This step is actually non-trivial to do in a unix environment when working with Office Documents, because Microsoft Office is a Windows suite. Using non-Windows software such as libreoffice as a substitute, results in poor quality PDF construction. 

Hence a visual-basic script has to be run which automates the opening of the pptx, and then saving the file as PDF.

<br>
### **2. Content Extraction**

#### **2.1 Page-Level Meta-data and Extraction of Document Elements**

Ths is definitely the meat of the work, and has the highest number of options that need to be explored.
* Open-source libraries (*PyMuPDF, PDFMiner, BeautifulSoup, Camelot, Spacy, Apache Tika, Tesseract...*)
* Proprietary solutions (*Amazon Textract, Google Cloud Document AI, Microsoft Azure Document Intelligence...*)
* Loading trained ML models (*TableTransformer, TabularOCR..*)
* Relying on prompting Multimodal LMs for extraction


There are MANY options for document-layout analysis ranging from free-options to paid-options. If the free-option is not working sufficiently well (after trying for a while), we should consider the paid option instead of over-engineering the free-option. The reasons is that 

(1) Document layout analysis is a very tedious and time consuming area, and it is better to get on with building the actual product. \\
(2) Even if you manage to finesse the open-source libraries, it is going to be brittle and hard to maintain, or just better handled by an official update. \\
(3) It would be extremely hard to justify annotating enough data to custom fine-tune a document analysis model.

Since we are working with enterprise documents, most likely the paid option to go with will be decided by who your cloud provider is. 

With many document sources, we would want to plug-and-play these solutions depending on how well they are performing on each of the unstructured document sources. As long as the library is well-supported, I dont have a good way to know what works *sufficiently well*, and trial and error on any new extraction method that comes along seems to be the way-of-life.

In my limited experience, Proprietary Solutions > Open-source Libraries > Multimodal Prompting > Off-the-shelf Pre-Trained ML models. The reason why trained ML models are ranked last is they probably overfit heavily to the dataset they published on. *Edit (2025)* Proprietary Solutions >  Multimodal Prompting > Open-source Libraries > Off-the-shelf Pre-Trained ML models



### **2.2 Visual sanity check**

Visual sanity checks are important when debugging or modifying a document layout solution. This means that whenever we change the code, the changes should reflect visually for a programmer to see. This is not unlike when frontend developers make changes and need to see what happens in the frontend. Typically I would construct two columns, one for the page extracted, and one for the resulting extraction in Json format. The bounding box of the image should be annotated for display. 

#### **2.3 Automated Testing**

Tests allow us to decide whether changing one of the layout or extraction modules helps or hurts out data pipeline, without having to do visual eye-balling.  This is simply a human constructed test file, of what we expect the extraction to look like (including the bounding box of images and diagrams).

<br>

### **3. A Flexible Document Extraction Pipeline**

A data pipeline (main.py) thus looks like the following
* Connector Object 
* DocumentFormatter Object
* MetaDataExtractor Object 
* DocumentExtractor Object 


**Page-Level Meta-data and Extraction of Document Elements** is really the meat of the work, hence we want to experiment with different document extraction and specialised modules rapidly.

I currently adopt the following organisation

<div style="font-size: 1em; line-height: 1.5em; padding: 10px; background-color: #f5f5f5; border-radius: 5px; overflow-x: auto;">
<pre>
.
├── data
│   ├── processed
│   │   ├── diagram
│   │   ├── image
│   │   ├── json
│   │   └── tables
│   ├── raw
│   └── standardised
├── src
│   ├── data_connectors
│   ├── document_extractors
│   ├── modules
│   │   ├── caption_extraction
│   │   ├── diagram_extraction
│   │   ├── OCR
│   │   ├── table_extraction
│   │   └── Translation
│   └── pipelines
│       ├── prepare_datasource1
│       └── prepare_datasource2
│           ├── main.py
│           ├── utils.py
│           └── visual_check.py
└── tests
    └── pipelines
        ├── prepare_datasource1_tests.py
        └── prepare_datasource2_tests.py
</pre>
</div>

Running the `main.py` in `src/pipelines/prepare_datasource1/main.py` should thus run Step 1 to 5 loading different choices of `document_extractor` and `modules`.

Each datasource has its own format and needs to be handled with its own customised Step 3 and 4. Step 3 and Step 4 can be bundled together, if using a proprietary (paid) document layout analyser, and separately if using roll-your-own solutions from open-source libraries.  

Tests should be tied to each data preparation pipelines. 

<br>


#### **Credits**
Keith Low who explored many things during his internship. In particular, 2.1 and 2.2
