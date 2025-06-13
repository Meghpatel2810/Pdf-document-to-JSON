# Adaptive Image Cropping OCR System
# Intelligent text-aware cropping with enhanced OCR accuracy

import cv2
import numpy as np
import easyocr
from PIL import Image
import os
import logging
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
import matplotlib.pyplot as plt
from scipy import ndimage
from skimage import measure, morphology
from datetime import datetime

logger = logging.getLogger('AdaptiveCroppingOCR')

class AdaptiveCroppingOCR:
    """
    Advanced OCR system that intelligently crops images into text-aware segments
    for improved accuracy while preserving text line integrity
    """
    
    def __init__(self, languages=['en'], gpu=True):
        """Initialize the adaptive cropping OCR system"""
        logger.info("Initializing Adaptive Cropping OCR System...")
        
        self.reader = easyocr.Reader(languages, gpu=gpu)
        
        # Configuration parameters
        self.config = {
            # Text detection parameters
            'min_text_height': 8,
            'max_text_height': 100,
            'line_spacing_tolerance': 0.3,  # 30% tolerance for line spacing
            'horizontal_merge_threshold': 20,
            'vertical_merge_threshold': 10,
            
            # Cropping parameters
            'crop_overlap': 10,  # Pixel overlap between crops
            'min_crop_width': 200,
            'min_crop_height': 50,
            'max_crop_width': 1000,
            'max_crop_height': 400,
            
            # Text line detection
            'text_density_threshold': 0.1,
            'blank_space_threshold': 20,
            'line_continuity_threshold': 0.7,
            
            # Quality thresholds
            'confidence_threshold': 0.4,
            'min_word_length': 2
        }
        
        logger.info("Adaptive Cropping OCR System initialized successfully")
    
    def process_image_with_adaptive_cropping(self, image, page_number=None, save_debug=True):
        """
        Main processing function that handles adaptive cropping and OCR
        """
        logger.info(f"Starting adaptive cropping OCR for page {page_number}")
        
        # Setup debug directory
        debug_dir = self._setup_debug_directory(page_number)
        
        # Convert to OpenCV format if needed
        opencv_image = self._convert_to_opencv(image)
        
        # Step 1: Analyze text layout and structure
        text_regions = self._analyze_text_layout(opencv_image, debug_dir, page_number)
        
        # Step 2: Create intelligent crops based on text analysis
        crop_regions = self._create_intelligent_crops(opencv_image, text_regions, debug_dir, page_number)
        
        # Step 3: Apply OCR to each crop with optimization
        crop_results = self._process_crops_with_ocr(opencv_image, crop_regions, debug_dir, page_number)
        
        # Step 4: Merge and reconstruct the full document text
        merged_result = self._merge_crop_results(crop_results, opencv_image.shape, page_number)
        
        # Step 5: Post-process and format final output
        final_result = self._post_process_merged_text(merged_result, page_number)
        
        logger.info(f"Adaptive cropping complete for page {page_number}: {len(final_result['formatted_text'])} characters")
        return final_result
    
    def _setup_debug_directory(self, page_number):
        """Setup debug directory structure"""
        try:
            from __main__ import DEBUG_DIR
            base_debug_dir = DEBUG_DIR
        except ImportError:
            base_debug_dir = "Debug"
        
        debug_dir = os.path.join(base_debug_dir, f"page_{page_number}_crops")
        os.makedirs(debug_dir, exist_ok=True)
        return debug_dir
    
    def _convert_to_opencv(self, image):
        """Convert PIL Image to OpenCV format"""
        if isinstance(image, Image.Image):
            return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        return image
    
    def _analyze_text_layout(self, image, debug_dir, page_number):
        """
        Analyze the image to understand text layout and identify text regions
        """
        logger.info(f"Analyzing text layout for page {page_number}")
        
        # Convert to grayscale for analysis
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Method 1: Connected Components Analysis
        text_regions_cc = self._find_text_regions_connected_components(gray, debug_dir)
        
        # Method 2: Horizontal/Vertical Projection Analysis
        text_regions_proj = self._find_text_regions_projection(gray, debug_dir)
        
        # Method 3: Contour-based Text Detection
        text_regions_contour = self._find_text_regions_contours(gray, debug_dir)
        
        # Method 4: Initial OCR-based region detection
        text_regions_ocr = self._find_text_regions_initial_ocr(image, debug_dir)
        
        # Combine and validate all methods
        combined_regions = self._combine_text_regions([
            text_regions_cc,
            text_regions_proj,
            text_regions_contour,
            text_regions_ocr
        ])
        
        # Save debug visualization
        self._save_text_regions_debug(image, combined_regions, debug_dir, "combined_text_regions")
        
        logger.info(f"Found {len(combined_regions)} text regions using combined analysis")
        return combined_regions
    
    def _find_text_regions_connected_components(self, gray, debug_dir):
        """Find text regions using connected components analysis"""
        
        # Apply threshold
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Morphological operations to connect text
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 25))
        
        # Connect horizontal text
        horizontal_lines = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, horizontal_kernel)
        
        # Find connected components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(horizontal_lines)
        
        regions = []
        for i in range(1, num_labels):  # Skip background
            x, y, w, h, area = stats[i]
            
            # Filter by size and aspect ratio
            if (w >= self.config['min_crop_width'] and 
                h >= self.config['min_text_height'] and
                h <= self.config['max_text_height'] * 3):  # Allow for multi-line regions
                
                regions.append({
                    'bbox': (x, y, x + w, y + h),
                    'method': 'connected_components',
                    'confidence': min(area / (w * h), 1.0),  # Density score
                    'area': area
                })
        
        # Save debug image
        debug_image = cv2.cvtColor(horizontal_lines, cv2.COLOR_GRAY2BGR)
        for region in regions:
            x1, y1, x2, y2 = region['bbox']
            cv2.rectangle(debug_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        cv2.imwrite(os.path.join(debug_dir, "connected_components_regions.png"), debug_image)
        
        return regions
    
    def _find_text_regions_projection(self, gray, debug_dir):
        """Find text regions using horizontal and vertical projection"""
        
        # Apply threshold
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Horizontal projection (sum of pixels in each row)
        horizontal_proj = np.sum(binary, axis=1)
        
        # Find text lines based on projection
        text_rows = []
        in_text = False
        start_row = 0
        
        for i, proj_val in enumerate(horizontal_proj):
            if proj_val > 0 and not in_text:
                # Start of text line
                in_text = True
                start_row = i
            elif proj_val == 0 and in_text:
                # End of text line
                in_text = False
                if i - start_row >= self.config['min_text_height']:
                    text_rows.append((start_row, i))
        
        # Handle case where text goes to end of image
        if in_text and len(horizontal_proj) - start_row >= self.config['min_text_height']:
            text_rows.append((start_row, len(horizontal_proj)))
        
        # Group nearby text rows into regions
        regions = []
        if text_rows:
            current_region_start = text_rows[0][0]
            current_region_end = text_rows[0][1]
            
            for start, end in text_rows[1:]:
                # Check if this row is close to the current region
                if start - current_region_end <= self.config['vertical_merge_threshold']:
                    # Extend current region
                    current_region_end = end
                else:
                    # Finish current region and start new one
                    regions.append({
                        'bbox': (0, current_region_start, gray.shape[1], current_region_end),
                        'method': 'projection',
                        'confidence': 0.8,
                        'text_rows': current_region_end - current_region_start
                    })
                    current_region_start = start
                    current_region_end = end
            
            # Add the last region
            regions.append({
                'bbox': (0, current_region_start, gray.shape[1], current_region_end),
                'method': 'projection',
                'confidence': 0.8,
                'text_rows': current_region_end - current_region_start
            })
        
        # Save debug visualization
        debug_image = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        
        # Draw horizontal projection
        proj_normalized = (horizontal_proj / np.max(horizontal_proj) * 100).astype(int)
        for i, val in enumerate(proj_normalized):
            cv2.line(debug_image, (gray.shape[1] - val, i), (gray.shape[1], i), (255, 0, 0), 1)
        
        # Draw regions
        for region in regions:
            x1, y1, x2, y2 = region['bbox']
            cv2.rectangle(debug_image, (x1, y1), (x2, y2), (0, 255, 255), 2)
        
        cv2.imwrite(os.path.join(debug_dir, "projection_regions.png"), debug_image)
        
        return regions
    
    def _find_text_regions_contours(self, gray, debug_dir):
        """Find text regions using contour detection"""
        
        # Apply threshold and morphological operations
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Morphological operations to connect nearby text
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3))
        morphed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        regions = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            
            # Filter contours by size
            if (w >= self.config['min_crop_width'] and 
                h >= self.config['min_text_height'] and
                h <= self.config['max_text_height'] * 2):
                
                # Calculate contour properties
                area = cv2.contourArea(contour)
                bbox_area = w * h
                fill_ratio = area / bbox_area if bbox_area > 0 else 0
                
                regions.append({
                    'bbox': (x, y, x + w, y + h),
                    'method': 'contours',
                    'confidence': fill_ratio,
                    'contour_area': area
                })
        
        # Save debug image
        debug_image = cv2.cvtColor(morphed, cv2.COLOR_GRAY2BGR)
        for region in regions:
            x1, y1, x2, y2 = region['bbox']
            cv2.rectangle(debug_image, (x1, y1), (x2, y2), (255, 0, 255), 2)
        
        cv2.imwrite(os.path.join(debug_dir, "contour_regions.png"), debug_image)
        
        return regions
    
    def _find_text_regions_initial_ocr(self, image, debug_dir):
        """Find text regions using initial OCR detection"""
        
        # Run OCR with low confidence to detect all possible text
        ocr_results = self.reader.readtext(image, width_ths=0.7, height_ths=0.7)
        
        regions = []
        for bbox, text, confidence in ocr_results:
            if confidence >= 0.2 and len(text.strip()) >= 2:  # Very low threshold for detection
                # Convert bbox to rectangle format
                x_coords = [point[0] for point in bbox]
                y_coords = [point[1] for point in bbox]
                
                x1, y1 = int(min(x_coords)), int(min(y_coords))
                x2, y2 = int(max(x_coords)), int(max(y_coords))
                
                regions.append({
                    'bbox': (x1, y1, x2, y2),
                    'method': 'initial_ocr',
                    'confidence': confidence,
                    'text_sample': text[:20]  # Store sample for debugging
                })
        
        # Group nearby OCR detections
        grouped_regions = self._group_nearby_regions(regions)
        
        # Save debug image
        debug_image = image.copy()
        for region in grouped_regions:
            x1, y1, x2, y2 = region['bbox']
            cv2.rectangle(debug_image, (x1, y1), (x2, y2), (0, 128, 255), 2)
        
        cv2.imwrite(os.path.join(debug_dir, "initial_ocr_regions.png"), debug_image)
        
        return grouped_regions
    
    def _group_nearby_regions(self, regions):
        """Group nearby regions together"""
        if not regions:
            return []
        
        grouped = []
        used = set()
        
        for i, region in enumerate(regions):
            if i in used:
                continue
            
            # Start a new group with this region
            group = [region]
            used.add(i)
            
            # Find nearby regions to group with
            for j, other_region in enumerate(regions):
                if j in used or i == j:
                    continue
                
                if self._regions_are_nearby(region['bbox'], other_region['bbox']):
                    group.append(other_region)
                    used.add(j)
            
            # Create combined region from group
            if len(group) > 1:
                combined_bbox = self._combine_bboxes([r['bbox'] for r in group])
                grouped_region = {
                    'bbox': combined_bbox,
                    'method': 'grouped_ocr',
                    'confidence': np.mean([r['confidence'] for r in group]),
                    'sub_regions': len(group)
                }
            else:
                grouped_region = group[0]
            
            grouped.append(grouped_region)
        
        return grouped
    
    def _regions_are_nearby(self, bbox1, bbox2):
        """Check if two bounding boxes are nearby enough to group"""
        x1a, y1a, x2a, y2a = bbox1
        x1b, y1b, x2b, y2b = bbox2
        
        # Check horizontal overlap or proximity
        horizontal_gap = max(0, max(x1a, x1b) - min(x2a, x2b))
        vertical_gap = max(0, max(y1a, y1b) - min(y2a, y2b))
        
        return (horizontal_gap <= self.config['horizontal_merge_threshold'] and 
                vertical_gap <= self.config['vertical_merge_threshold'])
    
    def _combine_bboxes(self, bboxes):
        """Combine multiple bounding boxes into one"""
        x1_min = min(bbox[0] for bbox in bboxes)
        y1_min = min(bbox[1] for bbox in bboxes)
        x2_max = max(bbox[2] for bbox in bboxes)
        y2_max = max(bbox[3] for bbox in bboxes)
        return (x1_min, y1_min, x2_max, y2_max)
    
    def _combine_text_regions(self, region_lists):
        """Combine text regions from multiple detection methods"""
        all_regions = []
        for regions in region_lists:
            all_regions.extend(regions)
        
        if not all_regions:
            return []
        
        # Remove overlapping regions (keep the one with highest confidence)
        filtered_regions = []
        for region in all_regions:
            should_add = True
            for existing in filtered_regions:
                if self._regions_overlap_significantly(region['bbox'], existing['bbox']):
                    if region['confidence'] > existing['confidence']:
                        # Replace existing with current
                        filtered_regions.remove(existing)
                        break
                    else:
                        # Don't add current
                        should_add = False
                        break
            
            if should_add:
                filtered_regions.append(region)
        
        # Sort regions by vertical position (top to bottom)
        filtered_regions.sort(key=lambda r: r['bbox'][1])
        
        return filtered_regions
    
    def _regions_overlap_significantly(self, bbox1, bbox2, threshold=0.5):
        """Check if two regions overlap significantly"""
        x1a, y1a, x2a, y2a = bbox1
        x1b, y1b, x2b, y2b = bbox2
        
        # Calculate intersection
        x1_int = max(x1a, x1b)
        y1_int = max(y1a, y1b)
        x2_int = min(x2a, x2b)
        y2_int = min(y2a, y2b)
        
        if x1_int >= x2_int or y1_int >= y2_int:
            return False
        
        intersection_area = (x2_int - x1_int) * (y2_int - y1_int)
        area1 = (x2a - x1a) * (y2a - y1a)
        area2 = (x2b - x1b) * (y2b - y1b)
        
        overlap_ratio = intersection_area / min(area1, area2)
        return overlap_ratio >= threshold
    
    def _create_intelligent_crops(self, image, text_regions, debug_dir, page_number):
        """Create intelligent crop regions based on text analysis"""
        logger.info(f"Creating intelligent crops for page {page_number}")
        
        if not text_regions:
            # Fallback: create grid-based crops
            return self._create_grid_crops(image)
        
        crops = []
        image_height, image_width = image.shape[:2]
        
        for i, region in enumerate(text_regions):
            x1, y1, x2, y2 = region['bbox']
            
            # Add padding and ensure bounds
            padding = self.config['crop_overlap']
            crop_x1 = max(0, x1 - padding)
            crop_y1 = max(0, y1 - padding)
            crop_x2 = min(image_width, x2 + padding)
            crop_y2 = min(image_height, y2 + padding)
            
            # Ensure minimum crop size
            crop_width = crop_x2 - crop_x1
            crop_height = crop_y2 - crop_y1
            
            if (crop_width >= self.config['min_crop_width'] and 
                crop_height >= self.config['min_crop_height']):
                
                crops.append({
                    'id': f'crop_{i:03d}',
                    'bbox': (crop_x1, crop_y1, crop_x2, crop_y2),
                    'source_region': region,
                    'width': crop_width,
                    'height': crop_height
                })
        
        # Save debug visualization
        self._save_crops_debug(image, crops, debug_dir, "intelligent_crops")
        
        logger.info(f"Created {len(crops)} intelligent crops for page {page_number}")
        return crops
    
    def _create_grid_crops(self, image):
        """Fallback: create grid-based crops"""
        crops = []
        image_height, image_width = image.shape[:2]
        
        crop_width = self.config['max_crop_width']
        crop_height = self.config['max_crop_height']
        overlap = self.config['crop_overlap']
        
        y = 0
        crop_id = 0
        
        while y < image_height:
            x = 0
            while x < image_width:
                x2 = min(x + crop_width, image_width)
                y2 = min(y + crop_height, image_height)
                
                crops.append({
                    'id': f'grid_crop_{crop_id:03d}',
                    'bbox': (x, y, x2, y2),
                    'source_region': {'method': 'grid_fallback'},
                    'width': x2 - x,
                    'height': y2 - y
                })
                
                crop_id += 1
                x += crop_width - overlap
                
                if x >= image_width:
                    break
            
            y += crop_height - overlap
            
            if y >= image_height:
                break
        
        return crops
    
    def _process_crops_with_ocr(self, image, crop_regions, debug_dir, page_number):
        """Process each crop with OCR optimization"""
        logger.info(f"Processing {len(crop_regions)} crops with OCR for page {page_number}")
        
        crop_results = []
        
        for i, crop_info in enumerate(crop_regions):
            logger.debug(f"Processing crop {i+1}/{len(crop_regions)}: {crop_info['id']}")
            
            # Extract crop from image
            x1, y1, x2, y2 = crop_info['bbox']
            crop_image = image[y1:y2, x1:x2]
            
            if crop_image.size == 0:
                continue
            
            # Apply preprocessing optimizations for this crop
            optimized_crops = self._optimize_crop_for_ocr(crop_image, crop_info['id'], debug_dir)
            
            # Run OCR on all optimized versions and choose best result
            best_result = self._run_ocr_on_crop_variants(optimized_crops, crop_info)
            
            if best_result:
                crop_results.append({
                    'crop_info': crop_info,
                    'ocr_result': best_result,
                    'position': (x1, y1, x2, y2)
                })
        
        logger.info(f"Successfully processed {len(crop_results)} crops for page {page_number}")
        return crop_results
    
    def _optimize_crop_for_ocr(self, crop_image, crop_id, debug_dir):
        """Apply various optimizations to a crop for better OCR"""
        
        optimized_crops = {}
        
        # Original
        optimized_crops['original'] = crop_image
        
        # Grayscale
        if len(crop_image.shape) == 3:
            gray = cv2.cvtColor(crop_image, cv2.COLOR_BGR2GRAY)
        else:
            gray = crop_image
        optimized_crops['grayscale'] = gray
        
        # Enhanced contrast
        enhanced = cv2.convertScaleAbs(gray, alpha=1.3, beta=20)
        optimized_crops['enhanced'] = enhanced
        
        # Gaussian blur + threshold
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        optimized_crops['threshold'] = thresh
        
        # Adaptive threshold
        adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        optimized_crops['adaptive'] = adaptive
        
        # Denoising
        denoised = cv2.fastNlMeansDenoising(gray)
        optimized_crops['denoised'] = denoised
        
        # Save debug images for first few crops
        if crop_id.endswith('000') or crop_id.endswith('001') or crop_id.endswith('002'):
            for method, img in optimized_crops.items():
                filename = f"{crop_id}_{method}.png"
                cv2.imwrite(os.path.join(debug_dir, filename), img)
        
        return optimized_crops
    
    def _run_ocr_on_crop_variants(self, optimized_crops, crop_info):
        """Run OCR on all crop variants and return the best result"""
        
        best_result = None
        best_score = 0
        
        for method, crop_image in optimized_crops.items():
            try:
                # Run OCR
                ocr_results = self.reader.readtext(crop_image)
                
                # Score this result
                score = self._score_crop_ocr_result(ocr_results)
                
                if score > best_score:
                    best_score = score
                    best_result = {
                        'method': method,
                        'ocr_results': ocr_results,
                        'score': score,
                        'text_elements': len([r for r in ocr_results if r[2] >= self.config['confidence_threshold']])
                    }
            
            except Exception as e:
                logger.warning(f"OCR failed for method {method}: {str(e)}")
                continue
        
        return best_result
    
    def _score_crop_ocr_result(self, ocr_results):
        """Score OCR results for a crop"""
        if not ocr_results:
            return 0.0
        
        total_confidence = 0.0
        valid_detections = 0
        total_text_length = 0
        
        for bbox, text, confidence in ocr_results:
            if confidence >= self.config['confidence_threshold'] and len(text.strip()) >= self.config['min_word_length']:
                total_confidence += confidence
                valid_detections += 1
                total_text_length += len(text.strip())
        
        if valid_detections == 0:
            return 0.0
        
        avg_confidence = total_confidence / valid_detections
        length_bonus = min(total_text_length / 100, 1.0)
        detection_bonus = min(valid_detections / 10, 1.0)
        
        return avg_confidence * 0.6 + length_bonus * 0.2 + detection_bonus * 0.2
    
    def _merge_crop_results(self, crop_results, image_shape, page_number):
        """Merge OCR results from all crops into a coherent document"""
        logger.info(f"Merging {len(crop_results)} crop results for page {page_number}")
        
        if not crop_results:
            return {'text_elements': [], 'formatted_text': ''}
        
        # Extract all text elements with their absolute positions
        all_text_elements = []
        
        for crop_result in crop_results:
            crop_bbox = crop_result['position']
            crop_x_offset, crop_y_offset = crop_bbox[0], crop_bbox[1]
            
            ocr_results = crop_result['ocr_result']['ocr_results']
            
            for bbox, text, confidence in ocr_results:
                if confidence >= self.config['confidence_threshold']:
                    # Convert relative bbox to absolute coordinates
                    abs_bbox = self._convert_bbox_to_absolute(bbox, crop_x_offset, crop_y_offset)
                    
                    text_element = {
                        'text': text.strip(),
                        'confidence': confidence,
                        'bbox': abs_bbox,
                        'crop_id': crop_result['crop_info']['id'],
                        'processing_method': crop_result['ocr_result']['method']
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
                    
                    all_text_elements.append(text_element)
        
        # Remove duplicate detections from overlapping crops
        deduplicated_elements = self._remove_duplicate_detections(all_text_elements)
        
        # Sort by reading order (top to bottom, left to right)
        sorted_elements = sorted(deduplicated_elements, key=lambda x: (x['center_y'], x['center_x']))
        
        return {
            'text_elements': sorted_elements,
            'total_elements': len(sorted_elements),
            'crops_processed': len(crop_results),
            'page_number': page_number
        }
    
    def _convert_bbox_to_absolute(self, relative_bbox, x_offset, y_offset):
        """Convert crop-relative bbox to absolute image coordinates"""
        absolute_bbox = []
        for point in relative_bbox:
            abs_x = point[0] + x_offset
            abs_y = point[1] + y_offset
            absolute_bbox.append([abs_x, abs_y])
        return absolute_bbox
    
    def _remove_duplicate_detections(self, text_elements):
        """Remove duplicate text detections from overlapping crops"""
        if not text_elements:
            return []
        
        # Group elements by spatial proximity and text similarity
        unique_elements = []
        used_indices = set()
        
        for i, element in enumerate(text_elements):
            if i in used_indices:
                continue
            
            # Find all similar elements
            similar_elements = [element]
            used_indices.add(i)
            
            for j, other_element in enumerate(text_elements[i+1:], i+1):
                if j in used_indices:
                    continue
                
                # Check if elements are duplicates
                if self._are_duplicate_detections(element, other_element):
                    similar_elements.append(other_element)
                    used_indices.add(j)
            
            # Choose the best element from similar ones
            best_element = max(similar_elements, key=lambda x: x['confidence'])
            unique_elements.append(best_element)
        
        logger.info(f"Removed {len(text_elements) - len(unique_elements)} duplicate detections")
        return unique_elements
    
    def _are_duplicate_detections(self, element1, element2):
        """Check if two text elements are likely duplicates"""
        # Check text similarity
        text_similarity = self._calculate_text_similarity(element1['text'], element2['text'])
        if text_similarity < 0.8:  # 80% text similarity threshold
            return False
        
        # Check spatial overlap
        spatial_overlap = self._calculate_spatial_overlap(element1, element2)
        if spatial_overlap < 0.5:  # 50% spatial overlap threshold
            return False
        
        return True
    
    def _calculate_text_similarity(self, text1, text2):
        """Calculate similarity between two text strings"""
        if not text1 or not text2:
            return 0.0
        
        # Simple character-based similarity
        text1_clean = text1.lower().strip()
        text2_clean = text2.lower().strip()
        
        if text1_clean == text2_clean:
            return 1.0
        
        # Calculate Levenshtein-like similarity
        from difflib import SequenceMatcher
        return SequenceMatcher(None, text1_clean, text2_clean).ratio()
    
    def _calculate_spatial_overlap(self, element1, element2):
        """Calculate spatial overlap between two elements"""
        # Get bounding rectangles
        x1a, y1a, x2a, y2a = element1['left'], element1['top'], element1['right'], element1['bottom']
        x1b, y1b, x2b, y2b = element2['left'], element2['top'], element2['right'], element2['bottom']
        
        # Calculate intersection
        x1_int = max(x1a, x1b)
        y1_int = max(y1a, y1b)
        x2_int = min(x2a, x2b)
        y2_int = min(y2a, y2b)
        
        if x1_int >= x2_int or y1_int >= y2_int:
            return 0.0
        
        intersection_area = (x2_int - x1_int) * (y2_int - y1_int)
        area1 = (x2a - x1a) * (y2a - y1a)
        area2 = (x2b - x1b) * (y2b - y1b)
        
        # Return overlap ratio relative to smaller element
        smaller_area = min(area1, area2)
        return intersection_area / smaller_area if smaller_area > 0 else 0.0
    
    def _post_process_merged_text(self, merged_result, page_number):
        """Post-process merged text results into final formatted output"""
        logger.info(f"Post-processing merged text for page {page_number}")
        
        text_elements = merged_result.get('text_elements', [])
        if not text_elements:
            return {
                'formatted_text': '',
                'statistics': {'total_elements': 0, 'page_number': page_number},
                'text_elements': []
            }
        
        # Group text elements into lines
        text_lines = self._group_elements_into_lines(text_elements)
        
        # Format lines with proper spacing
        formatted_lines = []
        for line in text_lines:
            formatted_line = self._format_text_line(line)
            if formatted_line.strip():
                formatted_lines.append(formatted_line)
        
        # Join lines and clean up formatting
        formatted_text = '\n'.join(formatted_lines)
        formatted_text = self._clean_formatted_text(formatted_text)
        
        # Calculate statistics
        statistics = self._calculate_final_statistics(merged_result, text_lines)
        
        return {
            'formatted_text': formatted_text,
            'text_lines': text_lines,
            'statistics': statistics,
            'metadata': {
                'page_number': page_number,
                'processing_method': 'adaptive_cropping',
                'total_crops': merged_result.get('crops_processed', 0),
                'processing_timestamp': datetime.now().isoformat()
            }
        }
    
    def _group_elements_into_lines(self, text_elements):
        """Group text elements into logical lines"""
        if not text_elements:
            return []
        
        # Sort elements by vertical position first
        sorted_elements = sorted(text_elements, key=lambda x: x['center_y'])
        
        lines = []
        current_line = [sorted_elements[0]]
        
        for element in sorted_elements[1:]:
            # Check if this element belongs to the current line
            last_element = current_line[-1]
            y_diff = abs(element['center_y'] - last_element['center_y'])
            
            # Calculate dynamic line threshold based on text height
            avg_height = (element['height'] + last_element['height']) / 2
            line_threshold = avg_height * self.config['line_spacing_tolerance']
            
            if y_diff <= line_threshold:
                # Same line - insert in correct horizontal position
                self._insert_element_in_line(current_line, element)
            else:
                # New line - finish current line and start new one
                lines.append(sorted(current_line, key=lambda x: x['center_x']))
                current_line = [element]
        
        # Add the last line
        if current_line:
            lines.append(sorted(current_line, key=lambda x: x['center_x']))
        
        return lines
    
    def _insert_element_in_line(self, line, element):
        """Insert element in correct horizontal position within a line"""
        inserted = False
        for i, line_element in enumerate(line):
            if element['center_x'] < line_element['center_x']:
                line.insert(i, element)
                inserted = True
                break
        if not inserted:
            line.append(element)
    
    def _format_text_line(self, line_elements):
        """Format a line of text elements with proper spacing"""
        if not line_elements:
            return ""
        
        # Sort elements by horizontal position
        sorted_elements = sorted(line_elements, key=lambda x: x['center_x'])
        
        formatted_line = ""
        prev_right = 0
        
        for i, element in enumerate(sorted_elements):
            text = element['text'].strip()
            if not text:
                continue
            
            current_left = element['left']
            
            if i > 0:
                # Calculate spacing based on gap between elements
                gap = current_left - prev_right
                avg_char_width = element['width'] / max(len(text), 1)
                
                if gap > avg_char_width * 6:  # Large gap - use multiple spaces/tabs
                    spaces = "\t\t"
                elif gap > avg_char_width * 3:  # Medium gap - use tab
                    spaces = "\t"
                elif gap > avg_char_width * 1.5:  # Small gap - use multiple spaces
                    spaces = "  "
                elif gap > avg_char_width * 0.5:  # Very small gap - use single space
                    spaces = " "
                else:  # No gap or overlap - no spacing
                    spaces = ""
                
                formatted_line += spaces
            
            formatted_line += text
            prev_right = element['right']
        
        return formatted_line
    
    def _clean_formatted_text(self, text):
        """Clean up formatted text"""
        if not text:
            return ""
        
        # Remove excessive whitespace
        lines = text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            # Clean up line spacing but preserve intentional formatting
            cleaned_line = line.rstrip()  # Remove trailing whitespace
            cleaned_line = ' '.join(cleaned_line.split())  # Normalize internal spacing
            
            if cleaned_line or (cleaned_lines and cleaned_lines[-1]):  # Keep some blank lines
                cleaned_lines.append(cleaned_line)
        
        # Remove excessive blank lines
        final_lines = []
        blank_count = 0
        
        for line in cleaned_lines:
            if line.strip():
                final_lines.append(line)
                blank_count = 0
            else:
                blank_count += 1
                if blank_count <= 2:  # Allow max 2 consecutive blank lines
                    final_lines.append(line)
        
        return '\n'.join(final_lines)
    
    def _calculate_final_statistics(self, merged_result, text_lines):
        """Calculate comprehensive statistics for the final result"""
        text_elements = merged_result.get('text_elements', [])
        
        # Basic counts
        total_elements = len(text_elements)
        total_lines = len(text_lines)
        total_crops = merged_result.get('crops_processed', 0)
        
        # Confidence statistics
        if text_elements:
            confidences = [elem['confidence'] for elem in text_elements]
            avg_confidence = sum(confidences) / len(confidences)
            min_confidence = min(confidences)
            max_confidence = max(confidences)
        else:
            avg_confidence = min_confidence = max_confidence = 0.0
        
        # Text statistics
        total_characters = sum(len(elem['text']) for elem in text_elements)
        total_words = sum(len(elem['text'].split()) for elem in text_elements)
        
        # Processing method distribution
        method_counts = {}
        for elem in text_elements:
            method = elem.get('processing_method', 'unknown')
            method_counts[method] = method_counts.get(method, 0) + 1
        
        return {
            'page_number': merged_result.get('page_number'),
            'total_elements': total_elements,
            'total_lines': total_lines,
            'total_crops_processed': total_crops,
            'total_characters': total_characters,
            'total_words': total_words,
            'confidence_stats': {
                'average': avg_confidence,
                'minimum': min_confidence,
                'maximum': max_confidence
            },
            'processing_methods': method_counts,
            'processing_timestamp': datetime.now().isoformat()
        }
    
    def _save_text_regions_debug(self, image, regions, debug_dir, filename):
        """Save debug visualization of text regions"""
        debug_image = image.copy()
        
        for region in regions:
            x1, y1, x2, y2 = region['bbox']
            # Color-code by method
            color = self._get_method_color(region['method'])
            cv2.rectangle(debug_image, (x1, y1), (x2, y2), color, 2)
            
            # Add method label
            cv2.putText(debug_image, region['method'], (x1, y1-5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        cv2.imwrite(os.path.join(debug_dir, f"{filename}.png"), debug_image)
    
    def _save_crops_debug(self, image, crops, debug_dir, filename):
        """Save debug visualization of crop regions"""
        debug_image = image.copy()
        
        for i, crop in enumerate(crops):
            x1, y1, x2, y2 = crop['bbox']
            cv2.rectangle(debug_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Add crop ID
            cv2.putText(debug_image, crop['id'], (x1, y1-5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        
        cv2.imwrite(os.path.join(debug_dir, f"{filename}.png"), debug_image)
    
    def _get_method_color(self, method):
        """Get color for debug visualization based on detection method"""
        colors = {
            'connected_components': (0, 255, 0),    # Green
            'projection': (255, 0, 0),              # Blue
            'contours': (0, 0, 255),                # Red
            'initial_ocr': (255, 255, 0),           # Cyan
            'grouped_ocr': (255, 0, 255),           # Magenta
            'grid_fallback': (128, 128, 128)        # Gray
        }
        return colors.get(method, (255, 255, 255))  # White default


# Integration class to combine with existing system
class EnhancedOCRProcessor:
    """
    Enhanced OCR processor that combines adaptive cropping with existing functionality
    """
    
    def __init__(self, languages=['en'], gpu=True):
        self.adaptive_cropping = AdaptiveCroppingOCR(languages, gpu)
        self.standard_processor = None  # Your existing processor
    
    def process_document_page(self, image, page_number=None, use_adaptive_cropping=True, save_debug=True):
        """
        Process a document page with enhanced OCR techniques
        """
        if use_adaptive_cropping:
            return self.adaptive_cropping.process_image_with_adaptive_cropping(
                image, page_number, save_debug
            )
        else:
            # Fallback to standard processing
            return self._standard_process(image, page_number, save_debug)
    
    def _standard_process(self, image, page_number, save_debug):
        """Fallback to standard OCR processing"""
        # Your existing OCR processing logic here
        pass


# Usage example and integration guide
def main_processing_example():
    """
    Example of how to integrate adaptive cropping into your existing pipeline
    """
    
    # Initialize the enhanced processor
    processor = EnhancedOCRProcessor(languages=['en'], gpu=True)
    
    # Process a PDF document
    import fitz  # PyMuPDF
    from PIL import Image
    from io import BytesIO
    
    pdf_path = r"Input/Apex1.pdf"
    doc = fitz.open(pdf_path)
    
    all_results = []
    
    for page_num in range(len(doc)):
        logger.info(f"Processing page {page_num + 1} of {len(doc)}")
        
        # Get page as high-resolution image
        page = doc.load_page(page_num)
        # Increase resolution for better OCR
        pix = page.get_pixmap(matrix=fitz.Matrix(3.0, 3.0))  # 3x zoom for better quality
        img_data = pix.tobytes("png")
        
        # Convert to PIL Image
        pil_image = Image.open(BytesIO(img_data))
        
        # Process with adaptive cropping
        result = processor.process_document_page(
            pil_image,
            page_number=page_num + 1,
            use_adaptive_cropping=True,
            save_debug=True
        )
        
        all_results.append(result)
        
        # Log progress
        stats = result.get('statistics', {})
        logger.info(f"Page {page_num + 1} processed: "
                   f"{stats.get('total_elements', 0)} elements, "
                   f"{stats.get('total_lines', 0)} lines, "
                   f"{stats.get('total_crops_processed', 0)} crops")
    
    doc.close()
    return all_results


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run example
    results = main_processing_example()