# Image Preprocessing Module
# Enhanced EasyOCR with layout preservation and unique page naming

import easyocr
import cv2
import numpy as np
import os
from PIL import Image
from datetime import datetime
import logging
import re
from typing import List, Dict, Tuple, Optional
from collections import defaultdict

# Get logger
logger = logging.getLogger('DocumentProcessor')

class EnhancedEasyOCRProcessor:
    """Enhanced EasyOCR with precise layout preservation and document structure recognition"""
    
    def __init__(self, languages=['en'], gpu=True):
        """Initialize Enhanced EasyOCR with configuration"""
        logger.info("Initializing Enhanced EasyOCR Processor...")
        self.reader = easyocr.Reader(languages, gpu=gpu)
        
        # Configuration for layout preservation
        self.line_threshold = 15  # Vertical threshold for grouping text into lines
        self.word_spacing_threshold = 20  # Horizontal threshold for word spacing
        self.section_break_threshold = 30  # Vertical threshold for section breaks
        self.confidence_threshold = 0.5  # Minimum confidence for text detection
        
        # Header identification keywords
        self.header_keywords = [
            'anthem', 'explanation', 'remittance', 'advice', 'office', 'street',
            'insurance', 'company', 'inc', 'co', 'llc', 'corp', 'health', 'life'
        ]
        
        # Document structure patterns
        self.section_patterns = {
            'provider_info': [r'provider\s+no:', r'check\s+no:', r'payer\s+id:', r'check\s+amount:', r'check\s+date:'],
            'patient_data': [r'patient\s+name:', r'claim\s+no:', r'insurance\s+id:', r'service\s+date', r'cpt'],
            'glossary': [r'glossary', r'adjustment\s+codes', r'co-\d+', r'pr-\d+'],
            'table_headers': [r'service\s+date', r'cpt', r'modifier', r'charge\s+amt', r'allowed\s+amt', r'paid\s+amt']
        }
        
        logger.info("Enhanced EasyOCR Processor initialized successfully")
    
    def extract_text_with_enhanced_layout(self, image, save_debug=True, page_number=None):
        """Extract text with enhanced layout preservation and unique page naming"""
        logger.info(f"Starting enhanced text extraction for page {page_number}...")
        
        # Import debug directory from main module
        try:
            from __main__ import DEBUG_DIR
            debug_dir = DEBUG_DIR
        except ImportError:
            debug_dir = "Debug"
            if not os.path.exists(debug_dir):
                os.makedirs(debug_dir)
        
        # Convert PIL Image to OpenCV format if needed
        if isinstance(image, Image.Image):
            opencv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        else:
            opencv_image = image
        
        image_height, image_width = opencv_image.shape[:2]
        logger.info(f"Processing image dimensions: {image_width}x{image_height}")
        
        # Try multiple preprocessing methods for best results with unique naming
        methods = self._get_preprocessing_methods(opencv_image, save_debug, debug_dir, page_number)
        
        best_result = None
        best_score = 0
        
        for method_name, processed_image in methods.items():
            logger.info(f"Trying {method_name} preprocessing for page {page_number}...")
            
            # Extract text with current method
            ocr_results = self.reader.readtext(processed_image)
            
            # Score this result
            score = self._score_ocr_results(ocr_results)
            logger.info(f"Page {page_number} - {method_name}: {len(ocr_results)} detections, score: {score:.2f}")
            
            if score > best_score:
                best_score = score
                best_result = {
                    'method': method_name,
                    'results': ocr_results,
                    'image': processed_image
                }
        
        if not best_result:
            logger.warning(f"No suitable OCR results found for page {page_number}")
            return {"formatted_text": "", "sections": {}, "statistics": {}}
        
        logger.info(f"Page {page_number} - Best method: {best_result['method']} with score: {best_score:.2f}")
        
        # Process the best results with enhanced layout preservation
        processed_data = self._process_enhanced_layout(
            best_result['results'], 
            image_width, 
            image_height,
            best_result['method'],
            page_number
        )
        
        return processed_data
    
    def _get_preprocessing_methods(self, image, save_debug, debug_dir, page_number=None):
        """Generate different preprocessing methods for OCR optimization with unique page naming"""
        methods = {}
        
        # Create page prefix for unique naming
        page_prefix = f"page{page_number}_" if page_number is not None else "page_unknown_"
        
        # Method 1: Original image
        methods['original'] = image
        if save_debug:
            filename = f"{page_prefix}method_original.png"
            cv2.imwrite(os.path.join(debug_dir, filename), image)
            logger.debug(f"Saved: {filename}")
        
        # Method 2: Grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        methods['grayscale'] = gray
        if save_debug:
            filename = f"{page_prefix}method_grayscale.png"
            cv2.imwrite(os.path.join(debug_dir, filename), gray)
            logger.debug(f"Saved: {filename}")
        
        # Method 3: Enhanced contrast
        enhanced = cv2.convertScaleAbs(gray, alpha=1.2, beta=10)
        methods['enhanced_contrast'] = enhanced
        if save_debug:
            filename = f"{page_prefix}method_enhanced_contrast.png"
            cv2.imwrite(os.path.join(debug_dir, filename), enhanced)
            logger.debug(f"Saved: {filename}")
        
        # Method 4: Gaussian blur + threshold
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        methods['threshold'] = thresh
        if save_debug:
            filename = f"{page_prefix}method_threshold.png"
            cv2.imwrite(os.path.join(debug_dir, filename), thresh)
            logger.debug(f"Saved: {filename}")
        
        # Method 5: Morphological operations
        kernel = np.ones((2, 2), np.uint8)
        morph = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
        methods['morphological'] = morph
        if save_debug:
            filename = f"{page_prefix}method_morphological.png"
            cv2.imwrite(os.path.join(debug_dir, filename), morph)
            logger.debug(f"Saved: {filename}")
        
        # Method 6: Adaptive threshold (additional method)
        adaptive_thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        methods['adaptive_threshold'] = adaptive_thresh
        if save_debug:
            filename = f"{page_prefix}method_adaptive_threshold.png"
            cv2.imwrite(os.path.join(debug_dir, filename), adaptive_thresh)
            logger.debug(f"Saved: {filename}")
        
        # Method 7: Denoising
        denoised = cv2.fastNlMeansDenoising(gray)
        methods['denoised'] = denoised
        if save_debug:
            filename = f"{page_prefix}method_denoised.png"
            cv2.imwrite(os.path.join(debug_dir, filename), denoised)
            logger.debug(f"Saved: {filename}")
        
        logger.info(f"Generated {len(methods)} preprocessing methods for page {page_number}")
        return methods
    
    def _score_ocr_results(self, ocr_results):
        """Score OCR results based on quality metrics"""
        if not ocr_results:
            return 0.0
        
        total_confidence = 0.0
        valid_detections = 0
        text_length = 0
        
        for bbox, text, confidence in ocr_results:
            if confidence >= self.confidence_threshold:
                total_confidence += confidence
                valid_detections += 1
                text_length += len(text.strip())
        
        if valid_detections == 0:
            return 0.0
        
        # Composite score: confidence + text amount + detection count
        avg_confidence = total_confidence / valid_detections
        length_score = min(text_length / 1000, 1.0)  # Normalize to 0-1
        detection_score = min(valid_detections / 100, 1.0)  # Normalize to 0-1
        
        return avg_confidence * 0.5 + length_score * 0.3 + detection_score * 0.2
    
    def _process_enhanced_layout(self, ocr_results, image_width, image_height, method_used, page_number=None):
        """Process OCR results with enhanced layout preservation"""
        logger.info(f"Processing enhanced layout structure for page {page_number}...")
        
        # Filter by confidence
        filtered_results = []
        for bbox, text, confidence in ocr_results:
            if confidence >= self.confidence_threshold and text.strip():
                filtered_results.append((bbox, text, confidence))
        
        logger.info(f"Page {page_number} - Filtered to {len(filtered_results)} high-confidence detections")
        
        if not filtered_results:
            return {"formatted_text": "", "sections": {}, "statistics": {}}
        
        # Convert to structured elements
        text_elements = self._create_text_elements(filtered_results)
        
        # Group into lines with enhanced spacing preservation
        lines = self._group_into_enhanced_lines(text_elements)
        
        # Identify document sections with improved logic
        sections = self._identify_enhanced_sections(lines)
        
        # Format with preserved spacing and organized headers
        formatted_output = self._format_enhanced_output(sections, image_width, method_used, page_number)
        
        return formatted_output
    
    def _create_text_elements(self, ocr_results):
        """Convert OCR results to structured text elements"""
        text_elements = []
        
        for bbox, text, confidence in ocr_results:
            # Calculate bounding box properties
            x_coords = [point[0] for point in bbox]
            y_coords = [point[1] for point in bbox]
            
            element = {
                'text': text.strip(),
                'confidence': confidence,
                'bbox': bbox,
                'left': min(x_coords),
                'right': max(x_coords),
                'top': min(y_coords),
                'bottom': max(y_coords),
                'center_x': sum(x_coords) / 4,
                'center_y': sum(y_coords) / 4,
                'width': max(x_coords) - min(x_coords),
                'height': max(y_coords) - min(y_coords)
            }
            text_elements.append(element)
        
        # Sort by Y-coordinate (top to bottom)
        text_elements.sort(key=lambda x: x['center_y'])
        return text_elements
    
    def _group_into_enhanced_lines(self, text_elements):
        """Group text elements into lines with enhanced logic"""
        if not text_elements:
            return []
        
        lines = []
        current_line = [text_elements[0]]
        
        for element in text_elements[1:]:
            # Check if this element belongs to the current line
            last_element = current_line[-1]
            y_diff = abs(element['center_y'] - last_element['center_y'])
            
            # Enhanced line grouping logic
            if y_diff <= self.line_threshold:
                # Same line - insert in correct X position
                self._insert_element_in_line(current_line, element)
            else:
                # New line
                lines.append(sorted(current_line, key=lambda x: x['center_x']))
                current_line = [element]
        
        # Add the last line
        if current_line:
            lines.append(sorted(current_line, key=lambda x: x['center_x']))
        
        return lines
    
    def _insert_element_in_line(self, line, element):
        """Insert element in correct position within a line"""
        inserted = False
        for i, line_element in enumerate(line):
            if element['center_x'] < line_element['center_x']:
                line.insert(i, element)
                inserted = True
                break
        if not inserted:
            line.append(element)
    
    def _identify_enhanced_sections(self, lines):
        """Enhanced section identification with better pattern matching"""
        logger.info("Identifying document sections with enhanced logic...")
        
        sections = {
            'header': [],
            'provider_info': [],
            'patient_data': [],
            'glossary': [],
            'footer': []
        }
        
        current_section = 'header'
        header_ended = False
        patient_section_started = False
        in_actual_glossary = False
        
        for line_idx, line in enumerate(lines):
            line_text = ' '.join([elem['text'] for elem in line])
            line_text_lower = line_text.lower()
            
            # Enhanced section detection logic
            if not header_ended:
                # Check for end of header - be more specific
                if any(pattern in line_text_lower for pattern in [
                    'provider no:', 'check no:', 'check amount:', 'payer id:'
                ]):
                    header_ended = True
                    current_section = 'provider_info'
                elif 'patient name:' in line_text_lower:
                    header_ended = True
                    current_section = 'patient_data'
                    patient_section_started = True
                else:
                    # Still in header - more restrictive header detection
                    if self._is_definite_header_content(line_text_lower):
                        current_section = 'header'
            
            # Post-header section detection
            if header_ended:
                # Check for actual glossary start
                if 'glossary:' in line_text_lower and 'adjustment codes' in line_text_lower:
                    current_section = 'glossary'
                    in_actual_glossary = True
                elif in_actual_glossary and (line_text_lower.startswith('pr-') or line_text_lower.startswith('co-')):
                    current_section = 'glossary'
                elif 'provider no:' in line_text_lower or 'check no:' in line_text_lower or 'payer id:' in line_text_lower:
                    current_section = 'provider_info'
                elif 'patient name:' in line_text_lower:
                    current_section = 'patient_data'
                    patient_section_started = True
                elif patient_section_started and self._is_service_line_data(line_text_lower):
                    current_section = 'patient_data'
                elif patient_section_started and self._is_table_header_or_data(line_text_lower):
                    current_section = 'patient_data'
                elif not in_actual_glossary and current_section == 'patient_data':
                    pass
            
            # Add line to appropriate section
            sections[current_section].append({
                'line_index': line_idx,
                'elements': line,
                'text': line_text,
                'y_position': min([elem['top'] for elem in line]) if line else 0,
                'section_confidence': self._calculate_section_confidence(line_text_lower, current_section)
            })
        
        # Log section distribution
        for section, lines_list in sections.items():
            logger.info(f"Section '{section}': {len(lines_list)} lines")
        
        return sections
    
    def _is_definite_header_content(self, text):
        """More restrictive header content detection"""
        header_indicators = [
            'anthem', 'insurance', 'company', 'health', 'life',
            'explanation', 'remittance', 'advice'
        ]
        
        if any(keyword in text for keyword in header_indicators):
            return True
        
        if re.search(r'\d+.*(?:way|street|avenue|drive|road|blvd)', text):
            return True
        
        if re.search(r'[a-z]+,\s*[a-z]{2}.*\d{5}', text):
            return True
        
        if 'office' in text and ('street' in text or re.search(r'\d+', text)):
            return True
        
        return False
    
    def _is_service_line_data(self, text):
        """Detect actual service line data"""
        service_patterns = [
            r'\d{2}/\d{2}/\d{4}\s+\d{5}',
            r'\d{2}/\d{2}/\d{4}\s+[A-Z]\d{4}',
            r'^\d{2}/\d{2}/\d{4}.*\d+\.\d{2}',
        ]
        
        return any(re.search(pattern, text) for pattern in service_patterns)
    
    def _is_table_header_or_data(self, text):
        """Enhanced detection for table headers and data"""
        table_headers = [
            'service date', 'cpt', 'modifier', 'charge amt', 'allowed amt', 
            'paid amt', 'patient res', 'writeoff', 'payment code'
        ]
        
        if any(header in text for header in table_headers):
            return True
        
        if 'claims total:' in text:
            return True
        
        if 'remark code:' in text:
            return True
        
        financial_pattern = r'\b\d+\.\d{2}\b'
        if len(re.findall(financial_pattern, text)) >= 2:
            return True
        
        return False
    
    def _calculate_section_confidence(self, text, section):
        """Calculate confidence score for section assignment"""
        confidence = 0.5
        
        if section in self.section_patterns:
            pattern_matches = sum(1 for pattern in self.section_patterns[section] 
                                if re.search(pattern, text, re.IGNORECASE))
            confidence += pattern_matches * 0.2
        
        return min(confidence, 1.0)
    
    def _format_enhanced_output(self, sections, image_width, method_used, page_number=None):
        """Format output with enhanced spacing and organized headers"""
        logger.info(f"Formatting enhanced output for page {page_number}...")
        
        output = {
            'formatted_text': '',
            'sections': sections,
            'statistics': {},
            'metadata': {}
        }
        
        formatted_lines = []
        
        # Enhanced header formatting
        if sections['header']:
            header_block = self._create_enhanced_header(sections['header'])
            formatted_lines.extend(header_block)
            formatted_lines.append("")
        
        # Format other sections with preserved spacing
        section_order = ['provider_info', 'patient_data', 'glossary', 'footer']
        
        for section_name in section_order:
            section_lines = sections.get(section_name, [])
            if section_lines:
                formatted_section = self._format_section_with_spacing(section_lines, section_name, image_width)
                formatted_lines.extend(formatted_section)
                formatted_lines.append("")
        
        output['formatted_text'] = '\n'.join(formatted_lines)
        output['statistics'] = self._calculate_enhanced_statistics(sections, method_used, page_number)
        output['metadata'] = {
            'page_number': page_number,
            'preprocessing_method': method_used,
            'line_threshold': self.line_threshold,
            'word_spacing_threshold': self.word_spacing_threshold,
            'section_break_threshold': self.section_break_threshold,
            'image_width': image_width,
            'processing_timestamp': datetime.now().isoformat()
        }
        
        logger.info(f"Enhanced formatting complete for page {page_number}: {len(output['formatted_text'])} characters")
        return output
    
    def _create_enhanced_header(self, header_lines):
        """Create intelligently organized header block"""
        if not header_lines:
            return []
        
        header_block = ["=" * 70, "DOCUMENT HEADER", "=" * 70]
        
        company_lines = []
        address_lines = []
        document_type_lines = []
        office_lines = []
        
        for line_data in header_lines:
            text = line_data['text'].strip()
            text_lower = text.lower()
            
            if any(keyword in text_lower for keyword in ['anthem', 'bc life', 'health ins co']):
                company_lines.append(text)
            elif re.search(r'\d{4}\s+[a-z]+.*way', text_lower):
                address_lines.append(text)
            elif re.search(r'cincinnati.*oh.*\d{5}', text_lower):
                address_lines.append(text)
            elif re.search(r'fresno.*ca.*\d{5}', text_lower):
                office_lines.append(text)
            elif 'explanation' in text_lower and 'remittance' in text_lower:
                document_type_lines.append(text)
            elif 'advice' in text_lower and len(text.split()) <= 3:
                document_type_lines.append(text)
            elif 'office' in text_lower or 'street' in text_lower:
                office_lines.append(text)
            elif text and not text.isspace() and len(text) > 2:
                if any(char.isdigit() for char in text) and len(text) < 20:
                    office_lines.append(text)
                elif len(text) > 20:
                    company_lines.append(text)
        
        if company_lines:
            header_block.extend(company_lines)
        
        if address_lines:
            header_block.extend(address_lines)
        
        if document_type_lines or office_lines:
            header_block.append("")
            if document_type_lines:
                header_block.extend(document_type_lines)
            if office_lines:
                header_block.extend(office_lines)
        
        header_block.append("=" * 70)
        return header_block
    
    def _format_section_with_spacing(self, section_lines, section_name, image_width):
        """Format section with preserved spacing"""
        formatted_section = []
        
        if section_name != 'header':
            section_title = section_name.replace('_', ' ').upper()
            formatted_section.append(f"{section_title} SECTION")
            formatted_section.append("-" * 50)
        
        for line_data in section_lines:
            formatted_line = self._format_line_with_enhanced_spacing(line_data['elements'], image_width)
            if formatted_line.strip():
                formatted_section.append(formatted_line)
        
        return formatted_section
    
    def _format_line_with_enhanced_spacing(self, elements, image_width):
        """Format line with enhanced spacing preservation"""
        if not elements:
            return ""
        
        sorted_elements = sorted(elements, key=lambda x: x['center_x'])
        formatted_line = ""
        prev_right = 0
        
        for i, element in enumerate(sorted_elements):
            text = element['text'].strip()
            current_left = element['left']
            
            if i > 0:
                gap = current_left - prev_right
                
                if gap > self.section_break_threshold * 2:
                    spaces = "\t\t"
                elif gap > self.section_break_threshold:
                    spaces = "\t"
                elif gap > self.word_spacing_threshold:
                    spaces = "    "
                elif gap > 10:
                    spaces = "  "
                elif gap > 5:
                    spaces = " "
                else:
                    spaces = ""
                
                formatted_line += spaces
            
            formatted_line += text
            prev_right = element['right']
        
        return formatted_line
    
    def _calculate_enhanced_statistics(self, sections, method_used, page_number=None):
        """Calculate comprehensive statistics"""
        total_elements = 0
        total_confidence = 0.0
        section_stats = {}
        
        for section_name, lines in sections.items():
            section_elements = 0
            section_confidence = 0.0
            
            for line in lines:
                line_elements = len(line['elements'])
                section_elements += line_elements
                
                for element in line['elements']:
                    section_confidence += element['confidence']
            
            total_elements += section_elements
            total_confidence += section_confidence
            
            section_stats[section_name] = {
                'lines': len(lines),
                'elements': section_elements,
                'avg_confidence': section_confidence / section_elements if section_elements > 0 else 0.0
            }
        
        return {
            'page_number': page_number,
            'total_elements': total_elements,
            'total_lines': sum(len(lines) for lines in sections.values()),
            'avg_confidence': total_confidence / total_elements if total_elements > 0 else 0.0,
            'preprocessing_method': method_used,
            'sections_found': [name for name, lines in sections.items() if lines],
            'section_statistics': section_stats,
            'processing_timestamp': datetime.now().isoformat()
        }


# Update the main processing function to pass page numbers
def process_document_with_page_numbers(pdf_path, debug_dir="Debug"):
    """Example of how to use the enhanced processor with page numbers"""
    processor = EnhancedEasyOCRProcessor()
    
    # Assuming you have a PDF processing loop
    import fitz  # PyMuPDF
    doc = fitz.open(pdf_path)
    
    all_results = []
    
    for page_num in range(len(doc)):
        logger.info(f"Processing page {page_num + 1} of {len(doc)}")
        
        # Get page as image
        page = doc.load_page(page_num)
        pix = page.get_pixmap(matrix=fitz.Matrix(2.0, 2.0))  # 2x zoom
        img_data = pix.tobytes("png")
        
        # Convert to PIL Image
        from io import BytesIO
        pil_image = Image.open(BytesIO(img_data))
        
        # Process with page number for unique naming
        result = processor.extract_text_with_enhanced_layout(
            pil_image, 
            save_debug=True, 
            page_number=page_num + 1  # 1-based page numbering
        )
        
        all_results.append(result)
    
    doc.close()
    return all_results