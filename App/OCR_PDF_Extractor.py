# Document Processing Application
# Simplified and modular approach

import fitz  # PyMuPDF
from pdf2image import convert_from_path
from PIL import Image
import os
import glob
from datetime import datetime
import logging
import re
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
import sys
from image_preprocessing import EnhancedEasyOCRProcessor

# =============================================================================
# CONFIGURATION - UPDATE THESE PATHS FOR YOUR SYSTEM
# =============================================================================

# Poppler path configuration (UPDATE THIS FOR YOUR SYSTEM)
# Windows example: r"C:\Poppler\poppler-24.08.0\Library\bin"
# Linux/Mac: Usually None (poppler should be in PATH)
POPPLER_PATH = r"C:\Poppler\poppler-24.08.0\Library\bin"  # UPDATE THIS PATH

# Project structure paths - corrected for your directory structure
# Since app.py is in App folder, we need to go up one level to project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_DIR = os.path.join(PROJECT_ROOT, "Input")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "Output")
DEBUG_DIR = os.path.join(PROJECT_ROOT, "Debug")
LOGS_DIR = os.path.join(PROJECT_ROOT, "Logs")
APP_DIR = os.path.dirname(os.path.abspath(__file__))  # Current App directory

# =============================================================================

# Create all directories if they don't exist
for directory in [INPUT_DIR, OUTPUT_DIR, DEBUG_DIR, LOGS_DIR]:
    if not os.path.exists(directory):
        os.makedirs(directory)

# Configure logging
def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(LOGS_DIR, f'document_processor_{datetime.now().strftime("%Y%m%d")}.log')),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger('DocumentProcessor')

# Initialize logger
logger = setup_logging()

def determine_pdf_type(pdf_path):
    """Determine if PDF is text-based or scanned"""
    logger.info(f"Analyzing PDF type for: {pdf_path}")
    
    try:
        doc = fitz.open(pdf_path)
        text_content = ""
        
        pages_to_check = min(3, len(doc))
        
        for page_num in range(pages_to_check):
            page = doc[page_num]
            text_content += page.get_text()
        
        doc.close()
        
        text_length = len(text_content.strip())
        pdf_type = "TEXT_BASED" if text_length > 100 else "SCANNED"
        
        logger.info(f"PDF type determined: {pdf_type} (text length: {text_length} characters)")
        return pdf_type
        
    except Exception as e:
        logger.error(f"Error determining PDF type: {str(e)}")
        raise

def extract_text_with_fitz(pdf_path):
    """Extract text directly from text-based PDFs"""
    logger.info("Starting text extraction using PyMuPDF")
    
    try:
        doc = fitz.open(pdf_path)
        full_text = ""
        layout_info = []
        total_pages = len(doc)
        
        for page_num in range(total_pages):
            page = doc[page_num]
            blocks = page.get_text("dict")
            page_text = page.get_text()
            
            full_text += f"\n=== PAGE {page_num + 1} ===\n"
            full_text += page_text + "\n"
            
            layout_info.append({
                'page': page_num,
                'blocks': blocks,
                'text': page_text
            })
        
        doc.close()
        logger.info(f"PyMuPDF extraction completed: {len(full_text)} total characters")
        return full_text, layout_info
        
    except Exception as e:
        logger.error(f"Error in PyMuPDF text extraction: {str(e)}")
        raise

def extract_text_with_enhanced_ocr(pdf_path):
    """Extract text from scanned PDFs using enhanced OCR with layout preservation"""
    logger.info("Starting enhanced OCR-based text extraction")
    
    try:
        logger.info("Converting PDF to images...")
        logger.info(f"Using Poppler path: {POPPLER_PATH if POPPLER_PATH else 'System PATH'}")
        
        # Convert PDF to images with proper poppler path handling
        if POPPLER_PATH and os.path.exists(POPPLER_PATH):
            images = convert_from_path(pdf_path, dpi=300, poppler_path=POPPLER_PATH)
            logger.info(f"Successfully converted PDF using Poppler at: {POPPLER_PATH}")
        else:
            # Try without poppler_path (for Linux/Mac or if poppler is in system PATH)
            try:
                images = convert_from_path(pdf_path, dpi=300)
                logger.info("Successfully converted PDF using system Poppler")
            except Exception as e:
                logger.error(f"Poppler path issue: {str(e)}")
                if POPPLER_PATH:
                    logger.error(f"Configured Poppler path: {POPPLER_PATH}")
                    logger.error("Please verify that the Poppler path is correct")
                else:
                    logger.error("Poppler not found in system PATH. Please install Poppler or set POPPLER_PATH")
                raise Exception(f"PDF to image conversion failed: {str(e)}")
        
        full_text = ""
        layout_info = []
        enhanced_ocr = EnhancedEasyOCRProcessor(gpu=True)
        total_pages = len(images)
        
        logger.info(f"Processing {total_pages} pages with Enhanced OCR")
        
        for page_num, image in enumerate(images):
            logger.info(f"Processing page {page_num + 1}/{total_pages}")
            
            # Save original for debugging
            original_path = os.path.join(DEBUG_DIR, f"page_{page_num + 1}_original.png")
            image.save(original_path)
            
            # Process with enhanced OCR
            page_results = enhanced_ocr.extract_text_with_enhanced_layout(image, save_debug=True)
            
            # Extract the formatted text
            page_text = page_results.get('formatted_text', '')
            page_statistics = page_results.get('statistics', {})
            
            logger.info(f"Page {page_num + 1}: {page_statistics.get('total_elements', 0)} elements, "
                       f"{len(page_text)} characters")
            
            # Save page-specific results
            page_text_path = os.path.join(DEBUG_DIR, f"page_{page_num + 1}_enhanced.txt")
            with open(page_text_path, 'w', encoding='utf-8') as f:
                f.write(page_text)
            
            # Add to full text
            full_text += f"\n=== PAGE {page_num + 1} ===\n"
            full_text += page_text + "\n"
            
            # Store layout info
            layout_info.append({
                'page': page_num,
                'text': page_text,
                'sections': page_results.get('sections', {}),
                'statistics': page_statistics,
                'metadata': page_results.get('metadata', {}),
                'preprocessing_method': page_statistics.get('preprocessing_method', 'unknown')
            })
        
        logger.info(f"Enhanced OCR extraction completed: {len(full_text)} total characters")
        return full_text, layout_info
        
    except Exception as e:
        logger.error(f"Error in enhanced OCR text extraction: {str(e)}")
        raise

def extract_text_from_pdf(pdf_path):
    """Main function to extract text from PDF with enhanced OCR"""
    logger.info(f"Starting PDF processing: {pdf_path}")
    
    if not os.path.exists(pdf_path):
        logger.error(f"PDF file not found: {pdf_path}")
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    
    try:
        pdf_type = determine_pdf_type(pdf_path)
        
        if pdf_type == "TEXT_BASED":
            logger.info("Using PyMuPDF for text-based PDF")
            return extract_text_with_fitz(pdf_path)
        else:
            logger.info("Using Enhanced OCR for scanned PDF")
            return extract_text_with_enhanced_ocr(pdf_path)
            
    except Exception as e:
        logger.error(f"Error in PDF text extraction: {str(e)}")
        raise

def save_extracted_text(extracted_text, layout_info, pdf_path):
    """Save extracted text with statistics to Debug folder"""
    logger.info("Saving extracted text to Debug folder")
    
    try:
        pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"{pdf_name}_extracted_{timestamp}.txt"
        output_path = os.path.join(DEBUG_DIR, output_filename)
        
        content = f"""
DOCUMENT PROCESSING REPORT
==========================
Source PDF: {pdf_path}
Processing Time: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
Total Pages: {len(layout_info)}

EXTRACTED TEXT:
===============
{extracted_text}

PROCESSING STATISTICS:
=====================
"""
        
        # Calculate statistics
        total_elements = 0
        total_confidence = 0.0
        preprocessing_methods = {}
        section_distribution = defaultdict(int)
        
        for page_info in layout_info:
            # Method tracking
            method = page_info.get('preprocessing_method', 'unknown')
            preprocessing_methods[method] = preprocessing_methods.get(method, 0) + 1
            
            # Statistics from processing
            stats = page_info.get('statistics', {})
            total_elements += stats.get('total_elements', 0)
            total_confidence += stats.get('avg_confidence', 0.0)
            
            # Section distribution
            sections = page_info.get('sections', {})
            for section_name, section_lines in sections.items():
                section_distribution[section_name] += len(section_lines)
        
        avg_confidence = total_confidence / len(layout_info) if layout_info else 0.0
        
        content += f"Total Text Elements: {total_elements}\n"
        content += f"Average OCR Confidence: {avg_confidence:.2f}\n"
        content += f"Total Characters Extracted: {len(extracted_text)}\n"
        
        if preprocessing_methods:
            content += f"\nPreprocessing Methods Used:\n"
            for method, count in preprocessing_methods.items():
                content += f"  {method}: {count} pages\n"
        
        if section_distribution:
            content += f"\nSection Distribution:\n"
            for section, line_count in section_distribution.items():
                content += f"  {section}: {line_count} lines\n"
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        logger.info(f"Extracted text saved to: {output_path}")
        return output_path
        
    except Exception as e:
        logger.error(f"Error saving extracted text: {str(e)}")
        raise

def process_document(pdf_path):
    """Process single document - simplified version"""
    logger.info(f"Starting document processing for: {pdf_path}")
    
    try:
        logger.info("Step 1: Text extraction from PDF")
        extracted_text, layout_info = extract_text_from_pdf(pdf_path)
        
        logger.info("Step 2: Saving extracted text")
        text_path = save_extracted_text(extracted_text, layout_info, pdf_path)
        
        logger.info("Processing completed successfully!")
        logger.info(f"Extracted text saved to: {text_path}")
        
        return {
            'text_path': text_path,
            'layout_info': layout_info,
            'extracted_text': extracted_text
        }
        
    except Exception as e:
        logger.error(f"Error in document processing: {str(e)}")
        raise

def get_pdf_files_from_input():
    """Get all PDF files from the Input directory"""
    pdf_files = glob.glob(os.path.join(INPUT_DIR, "*.pdf"))
    return pdf_files

def process_all_pdfs():
    """Process all PDF files in the Input directory"""
    pdf_files = get_pdf_files_from_input()
    
    if not pdf_files:
        logger.warning("No PDF files found in the Input directory")
        print("No PDF files found in the Input directory")
        return
    
    logger.info(f"Found {len(pdf_files)} PDF files to process")
    print(f"Found {len(pdf_files)} PDF files to process:")
    
    for pdf_file in pdf_files:
        print(f"  - {os.path.basename(pdf_file)}")
    
    results = []
    
    for i, pdf_file in enumerate(pdf_files, 1):
        print(f"\nProcessing file {i}/{len(pdf_files)}: {os.path.basename(pdf_file)}")
        logger.info(f"Processing file {i}/{len(pdf_files)}: {pdf_file}")
        
        try:
            result = process_document(pdf_file)
            results.append({
                'file': pdf_file,
                'status': 'success',
                'result': result
            })
            print(f"✓ Successfully processed: {os.path.basename(pdf_file)}")
            
        except Exception as e:
            logger.error(f"Failed to process {pdf_file}: {str(e)}")
            results.append({
                'file': pdf_file,
                'status': 'error',
                'error': str(e)
            })
            print(f"✗ Failed to process: {os.path.basename(pdf_file)} - {str(e)}")
    
    # Summary
    successful = len([r for r in results if r['status'] == 'success'])
    failed = len([r for r in results if r['status'] == 'error'])
    
    print(f"\n" + "=" * 70)
    print("PROCESSING SUMMARY")
    print("=" * 70)
    print(f"Total files: {len(pdf_files)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    
    if failed > 0:
        print("\nFailed files:")
        for result in results:
            if result['status'] == 'error':
                print(f"  - {os.path.basename(result['file'])}: {result['error']}")
    
    print(f"\nOutput files saved to: {DEBUG_DIR}")
    
    return results

def process_single_pdf(filename):
    """Process a single PDF file by name"""
    pdf_path = os.path.join(INPUT_DIR, filename)
    
    if not os.path.exists(pdf_path):
        logger.error(f"PDF file not found: {pdf_path}")
        print(f"Error: PDF file '{filename}' not found in Input directory")
        return None
    
    try:
        print(f"Processing: {filename}")
        result = process_document(pdf_path)
        print(f"✓ Successfully processed: {filename}")
        return result
        
    except Exception as e:
        logger.error(f"Failed to process {pdf_path}: {str(e)}")
        print(f"✗ Failed to process: {filename} - {str(e)}")
        return None

def main():
    """Main application entry point"""
    print("Document Processing Application")
    print("=" * 50)
    print(f"Project Root: {PROJECT_ROOT}")
    print(f"Input Directory: {INPUT_DIR}")
    print(f"Debug Directory: {DEBUG_DIR}")
    print(f"Logs Directory: {LOGS_DIR}")
    print("=" * 50)
    
    # Check CUDA availability
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✓ CUDA Available: GPU {torch.cuda.get_device_name(0)}")
        else:
            print("⚠ CUDA not available, using CPU for OCR")
    except ImportError:
        print("⚠ PyTorch not found, OCR will attempt to use available GPU")
    
    print("\nOptions:")
    print("1. Process all PDF files in Input directory")
    print("2. Process a specific PDF file")
    print("3. List PDF files in Input directory")
    print("4. Exit")
    
    while True:
        try:
            choice = input("\nEnter your choice (1-4): ").strip()
            
            if choice == '1':
                print("\nProcessing all PDF files...")
                process_all_pdfs()
                
            elif choice == '2':
                pdf_files = get_pdf_files_from_input()
                if not pdf_files:
                    print("No PDF files found in Input directory")
                    continue
                
                print("\nAvailable PDF files:")
                for i, pdf_file in enumerate(pdf_files, 1):
                    print(f"{i}. {os.path.basename(pdf_file)}")
                
                try:
                    file_choice = int(input("\nEnter file number: ")) - 1
                    if 0 <= file_choice < len(pdf_files):
                        selected_file = os.path.basename(pdf_files[file_choice])
                        process_single_pdf(selected_file)
                    else:
                        print("Invalid file number")
                except ValueError:
                    print("Please enter a valid number")
                
            elif choice == '3':
                pdf_files = get_pdf_files_from_input()
                if pdf_files:
                    print(f"\nFound {len(pdf_files)} PDF files:")
                    for i, pdf_file in enumerate(pdf_files, 1):
                        print(f"{i}. {os.path.basename(pdf_file)}")
                else:
                    print("No PDF files found in Input directory")
                
            elif choice == '4':
                print("Exiting application...")
                break
                
            else:
                print("Invalid choice. Please enter 1-4.")
                
        except KeyboardInterrupt:
            print("\n\nApplication interrupted by user")
            break
        except Exception as e:
            print(f"An error occurred: {str(e)}")
            logger.error(f"Application error: {str(e)}")

if __name__ == "__main__":
    main()