#Using chunking strategy to extract content from a website using crawl4ai 
#This is a simple example of how to use the crawl4ai library to extract content from a website using a chunking strategy
#The chunking strategy is a regex pattern that splits the text by paragraphs
#The extraction strategy is a LLMExtractionStrategy that uses the chunking strategy to extract the content
#The LLMConfig is the configuration for the LLM that will be used to extract the content
#The CrawlerRunConfig is the configuration for the crawler that will be used to crawl the website 
#The RegexChunking is the chunking strategy that will be used to split the text by paragraphs
#The LLMExtractionStrategy is the extraction strategy that will be used to extract the content from the website
#The AsyncWebCrawler is the crawler that will be used to crawl the website  

import asyncio  
from crawl4ai import AsyncWebCrawler, CrawlerRunConfig  
from crawl4ai.chunking_strategy import RegexChunking  
from crawl4ai.extraction_strategy import LLMExtractionStrategy  
from crawl4ai import LLMConfig  
  
async def main():  
    # Create a chunking strategy that splits text by paragraphs  
    chunking_strategy = RegexChunking(patterns=[r"\n\n"])  
      
    # Configure the extraction strategy to use the chunking strategy  
    extraction_strategy = LLMExtractionStrategy(  
        llm_config=LLMConfig(provider="openai/gpt-4o", api_token="provide your token here"),  
        schema={"type": "object", "properties": {"content": {"type": "string"}}},  
        instruction="Extract the main content from each chunk",  
        chunking_strategy=chunking_strategy,  
        chunk_token_threshold=2048  # Maximum tokens per chunk  
    )  
      
    # Configure the crawler  
    config = CrawlerRunConfig(  
        extraction_strategy=extraction_strategy,  
        word_count_threshold=10  # Minimum words per content block  
    )  
      
    # Run the crawler  
    async with AsyncWebCrawler() as crawler:  
        result = await crawler.arun("https://www.amazon.com/s?i=specialty-aps&bbn=16225007011&rh=n%3A16225007011%2Cn%3A193870011&ref=nav_em__nav_desktop_sa_intl_computer_components_0_2_7_3", config=config)  
          
        # Print the extracted content  
        print(result.extracted_content)  
  
if __name__ == "__main__":  
    asyncio.run(main())