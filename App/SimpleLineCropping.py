# Simple Line-by-Line Cropping Module
# Robust fallback version that always works

import cv2
import numpy as np
import easyocr
from PIL import Image
import os
import logging
from typing import List, Dict, Tuple, Optional
from datetime import datetime

logger = logging.getLogger('SimpleLineCropping')

class SimpleLineCroppingProcessor:
    """
    Simple and robust line-by-line cropping processor that:
    1. Enhances the full page image first
    2. Creates uniform horizontal strips (no complex detection)
    3. Applies OCR to each strip
    4. Combines results in reading order
    """
    
    def __init__(self, languages=['en'], gpu=True):
        """Initialize the simple line cropping processor"""
        logger.info("Initializing Simple Line Cropping Processor...")
        
        self.reader = easyocr.Reader(languages, gpu=gpu)
        
        # Simple configuration - no complex detection
        self.config = {
            'strip_height': 80,          # Height of each horizontal strip
            'strip_overlap': 10,         # Overlap between strips
            'min_strip_height': 20,      # Minimum strip height
            'confidence_threshold': 0.3,
            'min_word_length': 1
        }
        
        logger.info("Simple Line Cropping Processor initialized successfully")
    
    def process_image_with_simple_cropping(self, image, page_number=None, debug_dir=None):
        """
        Main processing function - enhance full image then create uniform strips
        """
        logger.info(f"Starting simple line cropping for page {page_number}")
        
        # Convert to OpenCV format
        opencv_image = self._convert_to_opencv(image)
        
        # Step 1: Enhance the full page image
        enhanced_image = self._enhance_full_page_image(opencv_image, debug_dir, page_number)
        
        # Step 2: Create uniform horizontal strips
        strips = self._create_uniform_strips(enhanced_image, debug_dir, page_number)
        
        # Step 3: Process each strip with OCR
        strip_results = self._process_strips_with_ocr(enhanced_image, strips, debug_dir, page_number)
        
        # Step 4: Combine results
        final_result = self._combine_strip_results(strip_results, opencv_image.shape, page_number)
        
        logger.info(f"Simple cropping complete for page {page_number}: {len(final_result.get('text_lines', []))} strips processed")
        return final_result
    
    def _convert_to_opencv(self, image):
        """Convert PIL Image to OpenCV format"""
        if isinstance(image, Image.Image):
            return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        return image
    
    def _enhance_full_page_image(self, image, debug_dir, page_number):
        """
        Enhance the full page image using the best method
        """
        logger.info(f"Enhancing full page image for page {page_number}")
        
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Try different enhancement methods
        enhancement_methods = {
            'enhanced_contrast': cv2.convertScaleAbs(gray, alpha=1.3, beta=15),
            'adaptive_threshold': cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2),
            'denoised': cv2.fastNlMeansDenoising(gray)
        }
        
        # Test each method quickly and pick the best
        best_method = 'enhanced_contrast'  # Safe default
        best_score = 0
        
        for method_name, enhanced_img in enhancement_methods.items():
            try:
                # Quick test on center crop
                h, w = enhanced_img.shape
                test_crop = enhanced_img[h//3:2*h//3, w//4:3*w//4]
                
                test_results = self.reader.readtext(test_crop)
                score = self._score_ocr_results(test_results)
                
                if score > best_score:
                    best_score = score
                    best_method = method_name
                    
            except Exception as e:
                logger.warning(f"OCR test failed for method {method_name}: {str(e)}")
                continue
        
        enhanced_image = enhancement_methods[best_method]
        logger.info(f"Selected enhancement method: {best_method} (score: {best_score:.2f})")
        
        # Save debug image
        if debug_dir:
            debug_filename = f"page_{page_number}_enhanced_{best_method}.png"
            cv2.imwrite(os.path.join(debug_dir, debug_filename), enhanced_image)
        
        return enhanced_image
    
    def _score_ocr_results(self, ocr_results):
        """Quick scoring of OCR results"""
        if not ocr_results:
            return 0.0
        
        total_confidence = sum(confidence for _, _, confidence in ocr_results)
        total_text_length = sum(len(text.strip()) for _, text, _ in ocr_results)
        
        avg_confidence = total_confidence / len(ocr_results)
        length_bonus = min(total_text_length / 50, 1.0)
        
        return avg_confidence * 0.7 + length_bonus * 0.3
    
    def _create_uniform_strips(self, enhanced_image, debug_dir, page_number):
        """
        Create uniform horizontal strips across the entire image
        """
        logger.info(f"Creating uniform horizontal strips for page {page_number}")
        
        image_height, image_width = enhanced_image.shape[:2]
        
        strips = []
        y = 0
        strip_id = 1
        
        while y < image_height:
            # Calculate strip boundaries
            y_end = min(y + self.config['strip_height'], image_height)
            
            # Ensure minimum strip height
            if y_end - y >= self.config['min_strip_height']:
                strips.append({
                    'strip_id': strip_id,
                    'y_start': y,
                    'y_end': y_end,
                    'height': y_end - y,
                    'bbox': (0, y, image_width, y_end)
                })
                strip_id += 1
            
            # Move to next strip with overlap
            y += self.config['strip_height'] - self.config['strip_overlap']
            
            # Prevent infinite loop
            if y >= image_height:
                break
        
        # Save debug visualization
        if debug_dir:
            self._save_strips_debug(enhanced_image, strips, debug_dir, page_number)
        
        logger.info(f"Created {len(strips)} uniform strips for page {page_number}")
        return strips
    
    def _save_strips_debug(self, image, strips, debug_dir, page_number):
        """Save debug visualization of strips"""
        debug_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) if len(image.shape) == 2 else image.copy()
        
        # Draw strip boundaries
        for strip in strips:
            x1, y1, x2, y2 = strip['bbox']
            cv2.rectangle(debug_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Add strip ID
            cv2.putText(debug_image, f"S{strip['strip_id']}", (x1 + 5, y1 + 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        debug_filename = f"page_{page_number}_uniform_strips.png"
        cv2.imwrite(os.path.join(debug_dir, debug_filename), debug_image)
    
    def _process_strips_with_ocr(self, enhanced_image, strips, debug_dir, page_number):
        """Process each strip with OCR"""
        logger.info(f"Processing {len(strips)} strips with OCR for page {page_number}")
        
        strip_results = []
        
        for strip_info in strips:
            strip_id = strip_info['strip_id']
            x1, y1, x2, y2 = strip_info['bbox']
            
            # Extract strip
            strip_image = enhanced_image[y1:y2, x1:x2]
            
            if strip_image.size == 0:
                logger.warning(f"Empty strip {strip_id}")
                continue
            
            # Save debug strip image for first few strips
            if debug_dir and strip_id <= 5:
                strip_filename = f"page_{page_number}_strip_{strip_id:02d}.png"
                cv2.imwrite(os.path.join(debug_dir, strip_filename), strip_image)
            
            # Apply OCR to strip
            try:
                ocr_results = self.reader.readtext(strip_image)
                
                # Process OCR results for this strip
                strip_text_elements = []
                for bbox, text, confidence in ocr_results:
                    if confidence >= self.config['confidence_threshold'] and len(text.strip()) >= self.config['min_word_length']:
                        # Convert bbox coordinates to absolute image coordinates
                        abs_bbox = self._convert_strip_bbox_to_absolute(bbox, x1, y1)
                        
                        text_element = {
                            'text': text.strip(),
                            'confidence': confidence,
                            'bbox': abs_bbox,
                            'strip_id': strip_id
                        }
                        
                        # Calculate position properties
                        x_coords = [point[0] for point in abs_bbox]
                        y_coords = [point[1] for point in abs_bbox]
                        
                        text_element.update({
                            'left': min(x_coords),
                            'right': max(x_coords),
                            'top': min(y_coords),
                            'bottom': max(y_coords),
                            'center_x': sum(x_coords) / 4,
                            'center_y': sum(y_coords) / 4,
                            'width': max(x_coords) - min(x_coords),
                            'height': max(y_coords) - min(y_coords)
                        })
                        
                        strip_text_elements.append(text_element)
                
                # Sort elements in this strip by horizontal position
                strip_text_elements.sort(key=lambda x: x['center_x'])
                
                strip_results.append({
                    'strip_info': strip_info,
                    'text_elements': strip_text_elements,
                    'strip_text': ' '.join([elem['text'] for elem in strip_text_elements]),
                    'total_elements': len(strip_text_elements),
                    'avg_confidence': np.mean([elem['confidence'] for elem in strip_text_elements]) if strip_text_elements else 0.0
                })
                
                if strip_text_elements:
                    logger.debug(f"Strip {strip_id}: {len(strip_text_elements)} elements, "
                               f"avg confidence: {np.mean([elem['confidence'] for elem in strip_text_elements]):.2f}")
                
            except Exception as e:
                logger.error(f"OCR failed for strip {strip_id}: {str(e)}")
                continue
        
        logger.info(f"Successfully processed {len(strip_results)} strips for page {page_number}")
        return strip_results
    
    def _convert_strip_bbox_to_absolute(self, strip_bbox, strip_x_offset, strip_y_offset):
        """Convert strip-relative bbox to absolute image coordinates"""
        absolute_bbox = []
        for point in strip_bbox:
            abs_x = point[0] + strip_x_offset
            abs_y = point[1] + strip_y_offset
            absolute_bbox.append([abs_x, abs_y])
        return absolute_bbox
    
    def _combine_strip_results(self, strip_results, image_shape, page_number):
        """Combine all strip results into final formatted output"""
        logger.info(f"Combining strip results for page {page_number}")
        
        if not strip_results:
            return {
                'formatted_text': '',
                'text_lines': [],
                'statistics': {
                    'total_strips': 0,
                    'total_elements': 0,
                    'avg_confidence': 0.0
                }
            }
        
        # Sort strips by vertical position
        strip_results.sort(key=lambda x: x['strip_info']['y_start'])
        
        # Create formatted text
        formatted_lines = []
        all_text_elements = []
        
        for strip_result in strip_results:
            strip_text = strip_result['strip_text']
            if strip_text.strip():
                formatted_lines.append(strip_text)
                all_text_elements.extend(strip_result['text_elements'])
        
        formatted_text = '\n'.join(formatted_lines)
        
        # Remove duplicate detections from overlapping strips
        deduplicated_elements = self._remove_duplicate_detections(all_text_elements)
        
        # Calculate statistics
        total_elements = len(deduplicated_elements)
        total_strips = len([sr for sr in strip_results if sr['strip_text'].strip()])
        avg_confidence = np.mean([elem['confidence'] for elem in deduplicated_elements]) if deduplicated_elements else 0.0
        
        statistics = {
            'page_number': page_number,
            'total_strips': total_strips,
            'total_elements': total_elements,
            'avg_confidence': avg_confidence,
            'strips_processed': len(strip_results),
            'processing_method': 'simple_uniform_strips'
        }
        
        return {
            'formatted_text': formatted_text,
            'text_lines': strip_results,
            'all_text_elements': deduplicated_elements,
            'statistics': statistics,
            'metadata': {
                'page_number': page_number,
                'processing_method': 'simple_uniform_strips',
                'image_shape': image_shape,
                'processing_timestamp': datetime.now().isoformat()
            }
        }
    
    def _remove_duplicate_detections(self, text_elements):
        """Remove duplicate text detections from overlapping strips"""
        if not text_elements:
            return []
        
        unique_elements = []
        used_indices = set()
        
        for i, element in enumerate(text_elements):
            if i in used_indices:
                continue
            
            # Find similar elements
            similar_elements = [element]
            used_indices.add(i)
            
            for j, other in enumerate(text_elements[i+1:], i+1):
                if j in used_indices:
                    continue
                
                if self._are_duplicate_detections(element, other):
                    similar_elements.append(other)
                    used_indices.add(j)
            
            # Choose best element
            best_element = max(similar_elements, key=lambda x: x['confidence'])
            unique_elements.append(best_element)
        
        logger.info(f"Removed {len(text_elements) - len(unique_elements)} duplicate detections")
        return unique_elements
    
    def _are_duplicate_detections(self, element1, element2):
        """Check if two elements are duplicates"""
        # Text similarity
        text_sim = self._text_similarity(element1['text'], element2['text'])
        if text_sim < 0.8:
            return False
        
        # Spatial overlap
        spatial_overlap = self._spatial_overlap(element1, element2)
        return spatial_overlap >= 0.5
    
    def _text_similarity(self, text1, text2):
        """Calculate text similarity"""
        if not text1 or not text2:
            return 0.0
        
        text1_clean = text1.lower().strip()
        text2_clean = text2.lower().strip()
        
        if text1_clean == text2_clean:
            return 1.0
        
        from difflib import SequenceMatcher
        return SequenceMatcher(None, text1_clean, text2_clean).ratio()
    
    def _spatial_overlap(self, element1, element2):
        """Calculate spatial overlap between elements"""
        x1a, y1a, x2a, y2a = element1['left'], element1['top'], element1['right'], element1['bottom']
        x1b, y1b, x2b, y2b = element2['left'], element2['top'], element2['right'], element2['bottom']
        
        # Intersection
        x1_int = max(x1a, x1b)
        y1_int = max(y1a, y1b)
        x2_int = min(x2a, x2b)
        y2_int = min(y2a, y2b)
        
        if x1_int >= x2_int or y1_int >= y2_int:
            return 0.0
        
        intersection = (x2_int - x1_int) * (y2_int - y1_int)
        area1 = (x2a - x1a) * (y2a - y1a)
        area2 = (x2b - x1b) * (y2b - y1b)
        
        smaller_area = min(area1, area2)
        return intersection / smaller_area if smaller_area > 0 else 0.0