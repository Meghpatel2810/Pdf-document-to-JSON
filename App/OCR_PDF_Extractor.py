# Document Processing Application
# Enhanced with Simple Line-by-Line Cropping for improved OCR accuracy

import fitz  
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


# Poppler path configuration 
POPPLER_PATH = r"C:\Poppler\poppler-24.08.0\Library\bin"  # UPDATE THIS PATH

# Project structure paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_DIR = os.path.join(PROJECT_ROOT, "Input")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "Output")
DEBUG_DIR = os.path.join(PROJECT_ROOT, "Debug")
LOGS_DIR = os.path.join(PROJECT_ROOT, "Logs")
APP_DIR = os.path.dirname(os.path.abspath(__file__))

# Default processing settings
USE_LINE_CROPPING = True 

# Create directories if they don't exist
for directory in [INPUT_DIR, OUTPUT_DIR, DEBUG_DIR, LOGS_DIR]:
    if not os.path.exists(directory):
        os.makedirs(directory)

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
                'text': page_text,
                'processing_method': 'fitz_text_extraction',
                'line_cropping_used': False
            })
        
        doc.close()
        logger.info(f"PyMuPDF extraction completed: {len(full_text)} total characters")
        return full_text, layout_info
        
    except Exception as e:
        logger.error(f"Error in PyMuPDF text extraction: {str(e)}")
        raise

def extract_text_with_enhanced_ocr(pdf_path, use_line_cropping=USE_LINE_CROPPING):
    """Extract text from scanned PDFs using enhanced OCR with optional line cropping"""
    processing_method = "Enhanced OCR with Line Cropping" if use_line_cropping else "Enhanced OCR (Standard)"
    logger.info(f"Starting {processing_method} text extraction")
    
    try:
        logger.info("Converting PDF to images...")
        logger.info(f"Using Poppler path: {POPPLER_PATH if POPPLER_PATH else 'System PATH'}")
        
        # Convert PDF to images
        if POPPLER_PATH and os.path.exists(POPPLER_PATH):
            images = convert_from_path(pdf_path, dpi=300, poppler_path=POPPLER_PATH)
            logger.info(f"Successfully converted PDF using Poppler at: {POPPLER_PATH}")
        else:
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
        enhanced_ocr = EnhancedEasyOCRProcessor(gpu=True, use_line_cropping=use_line_cropping)
        total_pages = len(images)
        
        logger.info(f"Processing {total_pages} pages with {processing_method}")
        
        for page_num, image in enumerate(images):
            logger.info(f"Processing page {page_num + 1}/{total_pages}")
            
            # Save original for debugging
            original_path = os.path.join(DEBUG_DIR, f"page_{page_num + 1}_original.png")
            image.save(original_path)
            
            # Process with enhanced OCR
            page_results = enhanced_ocr.extract_text_with_enhanced_layout(
                image, 
                save_debug=True, 
                page_number=page_num + 1
            )
            
            # Extract results
            page_text = page_results.get('formatted_text', '')
            page_statistics = page_results.get('statistics', {})
            page_metadata = page_results.get('metadata', {})
            
            # Log processing results
            total_elements = page_statistics.get('total_elements', 0)
            avg_confidence = page_statistics.get('avg_confidence', 0.0)
            line_cropping_used = page_metadata.get('line_cropping_used', False)
            preprocessing_method = page_statistics.get('preprocessing_method', 'unknown')
            
            logger.info(f"Page {page_num + 1}: {total_elements} elements, "
                       f"avg confidence: {avg_confidence:.2f}, "
                       f"line cropping: {'Yes' if line_cropping_used else 'No'}, "
                       f"method: {preprocessing_method}")
            
            if line_cropping_used:
                lines_detected = page_statistics.get('total_lines_detected', 0)
                logger.info(f"Page {page_num + 1}: {lines_detected} text lines detected and processed")
            
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
                'metadata': page_metadata,
                'preprocessing_method': preprocessing_method,
                'line_cropping_used': line_cropping_used,
                'processing_method': processing_method
            })
        
        logger.info(f"{processing_method} extraction completed: {len(full_text)} total characters")
        return full_text, layout_info
        
    except Exception as e:
        logger.error(f"Error in {processing_method} text extraction: {str(e)}")
        raise

def extract_text_from_pdf(pdf_path, force_ocr=False, use_line_cropping=USE_LINE_CROPPING):
    """Main function to extract text from PDF with enhanced OCR options"""
    logger.info(f"Starting PDF processing: {pdf_path}")
    logger.info(f"Line cropping enabled: {use_line_cropping}")
    logger.info(f"Force OCR mode: {force_ocr}")
    
    if not os.path.exists(pdf_path):
        logger.error(f"PDF file not found: {pdf_path}")
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    
    try:
        if force_ocr:
            logger.info("Forcing OCR processing (skipping text-based check)")
            return extract_text_with_enhanced_ocr(pdf_path, use_line_cropping)
        
        pdf_type = determine_pdf_type(pdf_path)
        
        if pdf_type == "TEXT_BASED":
            logger.info("Using PyMuPDF for text-based PDF")
            return extract_text_with_fitz(pdf_path)
        else:
            logger.info(f"Using Enhanced OCR for scanned PDF (line cropping: {use_line_cropping})")
            return extract_text_with_enhanced_ocr(pdf_path, use_line_cropping)
            
    except Exception as e:
        logger.error(f"Error in PDF text extraction: {str(e)}")
        raise

def save_extracted_text(extracted_text, layout_info, pdf_path):
    """Save extracted text with enhanced statistics to Debug folder"""
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
        
        # Calculate comprehensive statistics
        total_elements = 0
        total_confidence = 0.0
        preprocessing_methods = {}
        section_distribution = defaultdict(int)
        line_cropping_pages = 0
        total_lines_detected = 0
        
        for page_info in layout_info:
            # Method tracking
            method = page_info.get('processing_method', 'unknown')
            preprocessing_methods[method] = preprocessing_methods.get(method, 0) + 1
            
            # Line cropping tracking
            if page_info.get('line_cropping_used', False):
                line_cropping_pages += 1
                total_lines_detected += page_info.get('statistics', {}).get('total_lines_detected', 0)
            
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
        content += f"Pages Processed with Line Cropping: {line_cropping_pages}/{len(layout_info)}\n"
        
        if total_lines_detected > 0:
            content += f"Total Text Lines Detected: {total_lines_detected}\n"
        
        if preprocessing_methods:
            content += f"\nProcessing Methods Used:\n"
            for method, count in preprocessing_methods.items():
                content += f"  {method}: {count} pages\n"
        
        if section_distribution:
            content += f"\nSection Distribution:\n"
            for section, line_count in section_distribution.items():
                content += f"  {section}: {line_count} lines\n"
        
        # Add page-by-page breakdown
        content += f"\nPAGE-BY-PAGE BREAKDOWN:\n"
        content += "=" * 30 + "\n"
        
        for page_info in layout_info:
            page_num = page_info.get('page', 0) + 1
            stats = page_info.get('statistics', {})
            line_cropping_used = page_info.get('line_cropping_used', False)
            processing_method = page_info.get('processing_method', 'unknown')
            
            content += f"Page {page_num}:\n"
            content += f"  Processing Method: {processing_method}\n"
            content += f"  Text Elements: {stats.get('total_elements', 0)}\n"
            content += f"  Average Confidence: {stats.get('avg_confidence', 0.0):.2f}\n"
            content += f"  Line Cropping: {'Yes' if line_cropping_used else 'No'}\n"
            
            if line_cropping_used:
                lines = stats.get('total_lines_detected', 0)
                content += f"  Text Lines Detected: {lines}\n"
            
            content += "\n"
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        logger.info(f"Extracted text saved to: {output_path}")
        return output_path
        
    except Exception as e:
        logger.error(f"Error saving extracted text: {str(e)}")
        raise

def process_document(pdf_path, force_ocr=False, use_line_cropping=USE_LINE_CROPPING):
    """Process single document with enhanced options"""
    logger.info(f"Starting document processing for: {pdf_path}")
    logger.info(f"Force OCR: {force_ocr}, Line Cropping: {use_line_cropping}")
    
    try:
        logger.info("Step 1: Text extraction from PDF")
        extracted_text, layout_info = extract_text_from_pdf(pdf_path, force_ocr, use_line_cropping)
        
        logger.info("Step 2: Saving extracted text")
        text_path = save_extracted_text(extracted_text, layout_info, pdf_path)
        
        logger.info("Processing completed successfully!")
        logger.info(f"Extracted text saved to: {text_path}")
        
        return {
            'text_path': text_path,
            'layout_info': layout_info,
            'extracted_text': extracted_text,
            'line_cropping_used': use_line_cropping
        }
        
    except Exception as e:
        logger.error(f"Error in document processing: {str(e)}")
        raise

def get_pdf_files_from_input():
    """Get all PDF files from the Input directory"""
    pdf_files = glob.glob(os.path.join(INPUT_DIR, "*.pdf"))
    return pdf_files

def process_all_pdfs(force_ocr=False, use_line_cropping=USE_LINE_CROPPING):
    """Process all PDF files in the Input directory with enhanced options"""
    pdf_files = get_pdf_files_from_input()
    
    if not pdf_files:
        logger.warning("No PDF files found in the Input directory")
        print("No PDF files found in the Input directory")
        return
    
    logger.info(f"Found {len(pdf_files)} PDF files to process")
    print(f"Found {len(pdf_files)} PDF files to process:")
    print(f"Force OCR: {force_ocr}")
    print(f"Line Cropping: {use_line_cropping}")
    
    for pdf_file in pdf_files:
        print(f"  - {os.path.basename(pdf_file)}")
    
    results = []
    
    for i, pdf_file in enumerate(pdf_files, 1):
        print(f"\nProcessing file {i}/{len(pdf_files)}: {os.path.basename(pdf_file)}")
        logger.info(f"Processing file {i}/{len(pdf_files)}: {pdf_file}")
        
        try:
            result = process_document(pdf_file, force_ocr, use_line_cropping)
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
    print(f"Force OCR used: {force_ocr}")
    print(f"Line Cropping used: {use_line_cropping}")
    
    if failed > 0:
        print("\nFailed files:")
        for result in results:
            if result['status'] == 'error':
                print(f"  - {os.path.basename(result['file'])}: {result['error']}")
    
    print(f"\nOutput files saved to: {DEBUG_DIR}")
    
    return results

def process_single_pdf(filename, force_ocr=False, use_line_cropping=USE_LINE_CROPPING):
    """Process a single PDF file by name with enhanced options"""
    pdf_path = os.path.join(INPUT_DIR, filename)
    
    if not os.path.exists(pdf_path):
        logger.error(f"PDF file not found: {pdf_path}")
        print(f"Error: PDF file '{filename}' not found in Input directory")
        return None
    
    try:
        print(f"Processing: {filename}")
        print(f"Force OCR: {force_ocr}")
        print(f"Line Cropping: {use_line_cropping}")
        result = process_document(pdf_path, force_ocr, use_line_cropping)
        print(f"✓ Successfully processed: {filename}")
        return result
        
    except Exception as e:
        logger.error(f"Failed to process {pdf_path}: {str(e)}")
        print(f"✗ Failed to process: {filename} - {str(e)}")
        return None

def get_processing_options():
    """Get processing options from user"""
    print("\nProcessing Options:")
    print("1. Auto-detect (text-based vs scanned)")
    print("2. Force OCR (treat all PDFs as scanned)")
    
    while True:
        try:
            choice = input("Choose processing mode (1-2): ").strip()
            if choice == '1':
                force_ocr = False
                break
            elif choice == '2':
                force_ocr = True
                break
            else:
                print("Invalid choice. Please enter 1 or 2.")
        except KeyboardInterrupt:
            return None, None
    
    print("\nLine Cropping Options:")
    print("1. Enable line-by-line cropping (better accuracy, slower)")
    print("2. Use standard processing (faster)")
    
    while True:
        try:
            choice = input("Choose cropping mode (1-2): ").strip()
            if choice == '1':
                use_line_cropping = True
                break
            elif choice == '2':
                use_line_cropping = False
                break
            else:
                print("Invalid choice. Please enter 1 or 2.")
        except KeyboardInterrupt:
            return None, None
    
    return force_ocr, use_line_cropping

def main():
    """Main application entry point"""
    print("Enhanced Document Processing Application")
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
    
    print(f"\nDefault Settings:")
    print(f"  - Line Cropping: {'Enabled' if USE_LINE_CROPPING else 'Disabled'}")
    print(f"  - Processing Mode: Auto-detect")
    
    print("\nOptions:")
    print("1. Process all PDF files (with options)")
    print("2. Process a specific PDF file (with options)")
    print("3. Process all PDF files (default settings)")
    print("4. Process a specific PDF file (default settings)")
    print("5. List PDF files in Input directory")
    print("6. Exit")
    
    while True:
        try:
            choice = input("\nEnter your choice (1-6): ").strip()
            
            if choice == '1':
                print("\nProcessing all PDF files with custom options...")
                force_ocr, use_line_cropping = get_processing_options()
                if force_ocr is not None and use_line_cropping is not None:
                    process_all_pdfs(force_ocr, use_line_cropping)
                
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
                        force_ocr, use_line_cropping = get_processing_options()
                        if force_ocr is not None and use_line_cropping is not None:
                            process_single_pdf(selected_file, force_ocr, use_line_cropping)
                    else:
                        print("Invalid file number")
                except ValueError:
                    print("Please enter a valid number")
            
            elif choice == '3':
                print("\nProcessing all PDF files with default settings...")
                process_all_pdfs()
                
            elif choice == '4':
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
                
            elif choice == '5':
                pdf_files = get_pdf_files_from_input()
                if pdf_files:
                    print(f"\nFound {len(pdf_files)} PDF files:")
                    for i, pdf_file in enumerate(pdf_files, 1):
                        print(f"{i}. {os.path.basename(pdf_file)}")
                else:
                    print("No PDF files found in Input directory")
                
            elif choice == '6':
                print("Exiting application...")
                break
                
            else:
                print("Invalid choice. Please enter 1-6.")
                
        except KeyboardInterrupt:
            print("\n\nApplication interrupted by user")
            break
        except Exception as e:
            print(f"An error occurred: {str(e)}")
            logger.error(f"Application error: {str(e)}")

if __name__ == "__main__":
    main()