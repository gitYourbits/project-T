import spacy
import re
import json
import os
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
from dataclasses import dataclass, field
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import textstat  # Add textstat for better readability metrics

class ContentType(Enum):
    TITLE = "title"
    HEADING = "heading"
    CONCEPT = "concept"
    LIST_ITEM = "list_item"
    BULLET_POINT = "bullet_point"
    FORMULA = "formula"
    FACT = "fact"
    VISUAL_CUE = "visual_cue"
    TRANSITION = "transition"

class SceneType(Enum):
    INTRODUCTION = "introduction"
    DEFINITION = "definition"
    EXPLANATION = "explanation"
    DIAGRAM_REQUIRED = "diagram_required"
    IMAGE_REQUIRED = "image_required"
    SUMMARY = "summary"
    MAIN_CONTENT = "main_content"

@dataclass
class ContentBlock:
    content: str
    content_type: ContentType
    indentation_level: int
    parent_index: Optional[int] = None
    metadata: Dict = field(default_factory=dict)
    natural_text: Optional[str] = None

@dataclass
class Scene:
    id: int
    title: str
    blocks: List[ContentBlock]
    scene_type: SceneType
    states: List[str] = field(default_factory=list)
    features: List[str] = field(default_factory=list)
    visual_cues: List[str] = field(default_factory=list)
    natural_content: str = ""
    errors: List[Dict] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)
    
    def add_complexity_metadata(self, complexity_score, complexity_metadata: Dict):
        """Add complexity metadata to the scene."""
        self.metadata["complexity_score"] = complexity_score
        self.metadata["complexity_metrics"] = complexity_metadata

class ScriptParser:
    def __init__(self):
        # Initialize NLP components
        self._init_nlp_components()
        
        # Initialize error detection patterns
        self.error_patterns = {
            "grammar": [
                r"\b(is|are|was|were)\s+\w+ing\b",  # Incorrect continuous tense
                r"\b(a)\s+[aeiou]\w+",  # 'a' before vowel
                r"\b(an)\s+[^aeiou\s]\w+",  # 'an' before consonant
                r"\s+,",  # Space before comma
                r"\s+\.",  # Space before period
                r"\b(they|we|you|I)\s+(is|was)\b",  # Subject-verb agreement
                r"\b(he|she|it)\s+(are|were)\b"  # Subject-verb agreement
            ],
            "logic": [
                r"(?i)\b(however|but|although)\s+.{0,30}\s+(however|but|although)\b",  # Double contrast
                r"(?i)\b(therefore|thus|hence)\s+.{0,30}\s+(therefore|thus|hence)\b",  # Double conclusion
                r"(?i)\b(for example|such as)\s+.{0,30}\s+(for example|such as)\b"  # Double examples
            ],
            "incomplete": [
                r"(?i)(\.\.\.|…|etc\.|and so on)$",  # Ending with ellipsis
                r"(?i)\b(something|somehow|somewhere|someone)\b",  # Vague terms
                r"(?i)\b(many|several|some|few|various)\s+\w+$"  # Ending with indefinite quantifiers
            ]
        }
        
        # Scene type detection keywords
        self.scene_type_keywords = {
            SceneType.INTRODUCTION: ["introduction", "begin", "first", "start", "overview"],
            SceneType.DEFINITION: ["definition", "define", "concept", "term", "means"],
            SceneType.EXPLANATION: ["explain", "understand", "learn", "know", "comprehend"],
            SceneType.DIAGRAM_REQUIRED: ["diagram", "figure", "chart", "graph", "plot", "flowchart", "map"],
            SceneType.IMAGE_REQUIRED: ["image", "picture", "photo", "illustration", "visual", "see", "observe"],
            SceneType.SUMMARY: ["summary", "conclude", "conclusion", "end", "finally", "in summary"]
        }

    def _init_nlp_components(self):
        """Initialize all NLP components for parsing and analysis."""
        try:
            # Load spaCy model
            self.nlp = spacy.load("en_core_web_sm")
            
            # Initialize sentence transformer for semantic understanding
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            
            # Initialize text splitter for chunking
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )
            
            # Initialize embeddings for semantic search
            self.embeddings = HuggingFaceEmbeddings(
                model_name="all-MiniLM-L6-v2"
            )
            
        except Exception as e:
            print(f"Error initializing NLP components: {str(e)}")
            raise

    def extract_text_from_file(self, file_path: str) -> str:
        """Extract text from a file, supporting both .txt and .pdf formats."""
        try:
            file_ext = os.path.splitext(file_path)[1].lower()
            
            if file_ext == '.pdf':
                return self._extract_text_from_pdf(file_path)
            else:  # Default to text file
                with open(file_path, 'r', encoding='utf-8') as f:
                    return f.read()
                
        except Exception as e:
            print(f"Error reading file {file_path}: {str(e)}")
            raise

    def _extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text from a PDF file."""
        try:
            with open(pdf_path, 'rb') as file:
                reader = PdfReader(file)
                text = ""
                for page in reader.pages:
                    text += page.extract_text() + "\n"
                return text
        except Exception as e:
            print(f"Error extracting text from PDF: {str(e)}")
            raise

    def _preprocess_script(self, script: str) -> str:
        """Clean and normalize the input script."""
        # Remove extra whitespace between words while preserving line breaks
        script = re.sub(r'[ \t]+', ' ', script)
        
        # Normalize bullet points
        script = re.sub(r'^[\*\-•]\s*', '• ', script, flags=re.MULTILINE)
        
        # Normalize numbering
        script = re.sub(r'^\d+\.\s*', '', script, flags=re.MULTILINE)
        
        # Normalize roman numerals in parentheses
        script = re.sub(r'^\([ivx]+\)\s*', '', script, flags=re.MULTILINE)
        
        return script

    def segment_content(self, text: str) -> List[ContentBlock]:
        """Segment content into blocks with proper indentation and content type detection."""
        blocks = []
        lines = text.split('\n')
        
        # Track parent-child relationships
        current_indentation = 0
        parent_stack = []
        
        for i, line in enumerate(lines):
            if not line.strip():
                continue
                
            # Calculate indentation
            indentation = len(line) - len(line.lstrip())
            
            # Determine parent index based on indentation hierarchy
            parent_index = None
            if indentation > current_indentation:
                # This is a child of the previous block
                parent_index = len(blocks) - 1 if blocks else None
                parent_stack.append(parent_index)
            elif indentation < current_indentation:
                # Go back up the hierarchy
                steps_back = (current_indentation - indentation) // 2
                for _ in range(min(steps_back, len(parent_stack))):
                    parent_stack.pop()
                parent_index = parent_stack[-1] if parent_stack else None
            else:
                # Same level as before
                parent_index = parent_stack[-1] if parent_stack else None
            
            # Update current indentation
            current_indentation = indentation
            
            # Detect content type
            content_type = self._detect_content_type(line.strip())
            
            # Create content block
            block = ContentBlock(
                content=line.strip(),
                content_type=content_type,
                indentation_level=indentation // 2,  # Assuming 2 spaces per level
                parent_index=parent_index,
                metadata={}
            )
            
            # Also set natural text to be the cleaned content (no LLM hallucination)
            clean_text = line.strip()
            if content_type == ContentType.BULLET_POINT:
                clean_text = clean_text.replace('•', '').strip()
                if not clean_text.endswith('.'):
                    clean_text += '.'
            block.natural_text = clean_text
            
            blocks.append(block)
            
        return blocks

    def _detect_content_type(self, line: str) -> ContentType:
        """Detect the content type based on the line's pattern and structure."""
        # Check for markdown headings
        if re.match(r'^#{1,6}\s+', line):
            if line.startswith('# '):
                return ContentType.TITLE
            else:
                return ContentType.HEADING
        
        # Check for bullet points
        if line.startswith('•') or line.startswith('-'):
            return ContentType.BULLET_POINT
        
        # Check for visual cues
        if '[Visual:' in line or re.search(r'\[.*?(diagram|figure|image|map|picture|photo).*?\]', line, re.IGNORECASE):
            return ContentType.VISUAL_CUE
        
        # Check for formula
        if re.search(r'[=+\-*/]', line) and re.search(r'\d', line):
            return ContentType.FORMULA
        
        # Check for transitions
        transition_words = ['therefore', 'thus', 'consequently', 'first', 'second', 'lastly', 'finally']
        if any(line.lower().startswith(word) for word in transition_words):
            return ContentType.TRANSITION
        
        # Check for short factual statements
        if len(line.split()) < 15 and line.endswith('.'):
            return ContentType.FACT
        
        # Check for short headings
        if len(line.split()) < 7 and not line.endswith('.'):
            return ContentType.HEADING
            
        # Default to concept
        return ContentType.CONCEPT

    def _calculate_feature_importance(self, feature: str, blocks: List[ContentBlock], states: List[str]) -> float:
        """
        Calculate the importance of a feature using NLP techniques.
        Returns a score between 0.0 and 1.0 representing the importance.
        """
        # Baseline importance
        importance = 0.5
        
        # Create a document from the feature
        doc = self.nlp(feature)
        
        # 1. Keyword-based importance factors
        emphasis_keywords = [
            'important', 'critical', 'significant', 'key', 'main', 'essential', 'crucial', 'vital',
            # Enhanced list of emphasis keywords
            'fundamental', 'principal', 'primary', 'central', 'core', 'indispensable', 'imperative',
            'paramount', 'pivotal', 'predominant', 'prominent'
        ]
        if any(emphasis_word in feature.lower() for emphasis_word in emphasis_keywords):
            importance += 0.15
            
        # Check for explicit importance markers (format: @@importance:0.8)
        importance_marker = re.search(r'@@importance:(\d+(\.\d+)?)', feature)
        if importance_marker:
            try:
                explicit_importance = float(importance_marker.group(1))
                # Ensure it's within bounds
                explicit_importance = max(0.0, min(1.0, explicit_importance))
                # Use the explicit importance, but allow other factors to influence
                importance = 0.7 * explicit_importance + 0.3 * importance
                # Remove the marker from the feature
                feature = re.sub(r'@@importance:\d+(\.\d+)?', '', feature).strip()
            except ValueError:
                pass  # If conversion fails, continue with calculated importance
        
        # 2. Check for proper nouns, dates, numbers - indicates specific factual information
        has_proper_noun = any(token.pos_ == "PROPN" for token in doc)
        has_date = any(token.ent_type_ == "DATE" for token in doc if token.ent_type_)
        has_number = any(token.pos_ == "NUM" for token in doc) or re.search(r'\d+', feature)
        
        if has_proper_noun:
            importance += 0.1
        if has_date:
            importance += 0.1
        if has_number:
            importance += 0.1
        
        # 3. Length-based importance (longer might be more detailed/important)
        word_count = len(doc)
        if word_count > 10:  # Longer points might be more detailed
            importance += 0.05
        
        # 4. Check for technical terms or domain-specific language
        technical_terms = [token.text for token in doc if token.is_alpha and len(token.text) > 5 and not token.is_stop]
        if len(technical_terms) > 0:
            importance += min(0.1, 0.02 * len(technical_terms))
        
        # 5. Check for semantic relationship to states
        if states:
            state_text = " ".join(states)
            state_doc = self.nlp(state_text)
            
            # Check if feature mentions any of the states directly
            if any(state.lower() in feature.lower() for state in states):
                importance += 0.1
            
            # Or if it contains semantically similar concepts
            # Simplified semantic similarity using spaCy
            similarity = max([feature_token.similarity(state_token) 
                             for feature_token in doc if not feature_token.is_stop
                             for state_token in state_doc if not state_token.is_stop], default=0)
            
            importance += similarity * 0.1
        
        # 6. Presence of causal or reasoning language
        causal_indicators = ['because', 'therefore', 'thus', 'hence', 'due to', 'as a result', 'leads to', 'causes']
        if any(indicator in feature.lower() for indicator in causal_indicators):
            importance += 0.1
            
        # 7. NEW: Check for educational value indicators
        educational_indicators = ['learn', 'understand', 'concept', 'knowledge', 'skill', 'comprehend', 'apply', 
                                 'study', 'practice', 'master', 'grasp', 'foundation', 'basics', 'principle']
        if any(indicator in feature.lower() for indicator in educational_indicators):
            importance += 0.12
            
        # 8. NEW: Detect if this is a prerequisite or foundational concept
        prerequisite_indicators = ['prerequisite', 'required', 'necessary', 'need to know', 'foundation', 'basis',
                                  'before', 'first', 'initially', 'begin with', 'start with']
        if any(indicator in feature.lower() for indicator in prerequisite_indicators):
            importance += 0.15
            
        # Ensure importance is within range [0, 1]
        return min(max(importance, 0.0), 1.0)

    def _extract_states_and_features(self, blocks: List[ContentBlock]) -> Tuple[List[str], List[Dict]]:
        """Extract states and features from content blocks."""
        states = []
        features = []
        
        # Look for state patterns
        state_pattern = r"States:\s*(.*?)(?=\n|$)"
        state_list_pattern = r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b"
        
        for block in blocks:
            content = block.content
            
            # Extract states
            state_match = re.search(state_pattern, content, re.IGNORECASE)
            if state_match:
                state_text = state_match.group(1)
                # Split by commas, 'and', or other separators
                state_parts = re.split(r',|\sand\s|&', state_text)
                for part in state_parts:
                    part = part.strip()
                    if part:
                        states.append(part)
            else:
                # Look for capitalized words that might be state names
                state_matches = re.finditer(state_list_pattern, content)
                for match in state_matches:
                    potential_state = match.group(1)
                    # Check if it looks like a state name (heuristic)
                    if len(potential_state.split()) <= 3:  # Most state names are 1-3 words
                        states.append(potential_state)
            
            # Extract features from bullet points
            if block.content_type == ContentType.BULLET_POINT:
                clean_content = block.content.replace('•', '').strip()
                features.append(clean_content)
        
        # Remove duplicates
        states = list(set(states))
        features = list(set(features))
        
        # Calculate importance for each feature
        weighted_features = []
        for feature in features:
            importance = self._calculate_feature_importance(feature, blocks, states)
            weighted_features.append({
                "text": feature,
                "importance": importance
            })
            
        # Sort features by importance (descending)
        weighted_features.sort(key=lambda x: x["importance"], reverse=True)
        
        return states, weighted_features

    def _extract_visual_cues(self, blocks: List[ContentBlock]) -> List[str]:
        """Extract visual cues from content blocks."""
        visual_cues = []
        
        # Pattern for bracketed visual instructions
        visual_pattern = r'\[(Visual:.*?|.*?diagram.*?|.*?figure.*?|.*?image.*?|.*?map.*?|.*?picture.*?|.*?photo.*?)\]'
        
        # Pattern for explicit visual cue markers (enhanced with new marker syntax)
        explicit_markers = [
            r'@@diagram:\s*(.*?)(?=\n|$)',
            r'@@image:\s*(.*?)(?=\n|$)',
            r'@@figure:\s*(.*?)(?=\n|$)',
            r'@@chart:\s*(.*?)(?=\n|$)',
            r'@@map:\s*(.*?)(?=\n|$)',
            r'@@infographic:\s*(.*?)(?=\n|$)',
            r'@@animation:\s*(.*?)(?=\n|$)',
            r'@@video:\s*(.*?)(?=\n|$)',
            r'@@3d:\s*(.*?)(?=\n|$)'
        ]
        
        for block in blocks:
            content = block.content
            
            # Extract explicit visual cues
            if block.content_type == ContentType.VISUAL_CUE:
                visual_cues.append(content)
            else:
                # Look for visual cue patterns
                matches = re.finditer(visual_pattern, content, re.IGNORECASE)
                for match in matches:
                    visual_cues.append(match.group(0))
                
                # Look for explicit markers (@@diagram:, @@image:, etc.)
                for pattern in explicit_markers:
                    matches = re.finditer(pattern, content, re.IGNORECASE)
                    for match in matches:
                        cue_text = match.group(1).strip()
                        marker_type = pattern.split(':')[0].replace(r'@@', '').strip()
                        
                        visual_cues.append(f"[{marker_type.capitalize()}: {cue_text}]")
                        block.metadata["visual_type"] = marker_type
                        # Store the original marker for reference
                        block.metadata["visual_marker"] = f"@@{marker_type}: {cue_text}"
                
                # Also check for mentions of visual elements in the text
                visual_keywords = {
                    'diagram': 'Diagram',
                    'figure': 'Figure', 
                    'image': 'Image',
                    'map': 'Map',
                    'picture': 'Picture',
                    'photo': 'Photo',
                    'illustration': 'Illustration',
                    'chart': 'Chart',
                    'graph': 'Graph',
                    'flowchart': 'Flowchart',
                    'infographic': 'Infographic',
                    'visualization': 'Visualization',
                    'animation': 'Animation',
                    'video': 'Video',
                    '3d model': '3D Model'
                }
                
                for keyword, cue_type in visual_keywords.items():
                    if keyword in content.lower():
                        # Get the sentence containing the keyword
                        sentences = re.split(r'[.!?]', content)
                        for sentence in sentences:
                            if keyword in sentence.lower():
                                visual_cues.append(f"[{cue_type}: {sentence.strip()}]")
        
        # Remove duplicates
        return list(set(visual_cues))

    def _determine_scene_type(self, blocks: List[ContentBlock], title: str) -> SceneType:
        """Determine the scene type based on content and title."""
        # Convert all content to one text for analysis
        all_content = title + " " + " ".join([block.content for block in blocks])
        all_content_lower = all_content.lower()
        
        # Check for explicit visual requirements in block metadata
        for block in blocks:
            if "visual_type" in block.metadata:
                visual_type = block.metadata["visual_type"]
                if visual_type in ["diagram", "chart", "flowchart", "map"]:
                    return SceneType.DIAGRAM_REQUIRED
                elif visual_type in ["image", "picture", "photo", "illustration"]:
                    return SceneType.IMAGE_REQUIRED
        
        # Check for explicit markers in content
        diagram_markers = ["@@diagram:", "[diagram:", "[figure:"]
        image_markers = ["@@image:", "[image:", "[picture:", "[photo:"]
        
        if any(marker in all_content_lower for marker in diagram_markers):
            return SceneType.DIAGRAM_REQUIRED
        
        if any(marker in all_content_lower for marker in image_markers):
            return SceneType.IMAGE_REQUIRED
        
        # Check for visual cues in content blocks
        for block in blocks:
            if block.content_type == ContentType.VISUAL_CUE:
                content_lower = block.content.lower()
                
                # Check for diagram-related terms
                for diagram_keyword in self.scene_type_keywords[SceneType.DIAGRAM_REQUIRED]:
                    if diagram_keyword in content_lower:
                        return SceneType.DIAGRAM_REQUIRED
                
                # Check for image-related terms
                for image_keyword in self.scene_type_keywords[SceneType.IMAGE_REQUIRED]:
                    if image_keyword in content_lower:
                        return SceneType.IMAGE_REQUIRED
        
        # Check if title indicates this is an introduction
        if any(keyword in title.lower() for keyword in self.scene_type_keywords[SceneType.INTRODUCTION]):
            return SceneType.INTRODUCTION
            
        # Check if title indicates this is a summary or conclusion
        if any(keyword in title.lower() for keyword in self.scene_type_keywords[SceneType.SUMMARY]):
            return SceneType.SUMMARY
            
        # Check for definition markers
        definition_indicators = ["define", "definition", "refers to", "means", "is a", "is an", "is the"]
        if any(indicator in all_content_lower for indicator in definition_indicators):
            return SceneType.DEFINITION
        
        # Check for explanation markers (more complex content analysis)
        explanation_indicators = ["explain", "understand", "works by", "process of", "how to", "reasons", "because"]
        if any(indicator in all_content_lower for indicator in explanation_indicators):
            return SceneType.EXPLANATION
        
        # Default to main content
        return SceneType.MAIN_CONTENT

    def _detect_and_fix_errors(self, blocks: List[ContentBlock]) -> List[ContentBlock]:
        """Detect and fix errors in content blocks."""
        for block in blocks:
            text = block.natural_text or block.content
            
            # Only process blocks that are not headings or titles
            if block.content_type not in [ContentType.TITLE, ContentType.HEADING]:
                # Fix grammatical issues
                # 1. Add missing periods
                if not text.endswith('.') and not text.endswith('!') and not text.endswith('?'):
                    text = text + '.'
                
                # 2. Fix 'a' before vowels -> 'an'
                text = re.sub(r'\b(a)\s+([aeiou])', r'an \2', text, flags=re.IGNORECASE)
                
                # 3. Fix 'an' before consonants -> 'a'
                text = re.sub(r'\b(an)\s+([^aeiou\s])', r'a \2', text, flags=re.IGNORECASE)
                
                # 4. Add missing subjects (for bullet points)
                if block.content_type == ContentType.BULLET_POINT:
                    doc = self.nlp(text)
                    has_subject = any(token.dep_ == 'nsubj' for token in doc)
                    
                    if not has_subject:
                        # For bullet points, let's use the parent context as subject if available
                        if block.parent_index is not None and block.parent_index < len(blocks):
                            parent_block = blocks[block.parent_index]
                            parent_text = parent_block.content
                            
                            # Extract a potential subject from the parent
                            parent_doc = self.nlp(parent_text)
                            potential_subjects = []
                            
                            for token in parent_doc:
                                if token.pos_ in ['NOUN', 'PROPN'] and token.dep_ in ['nsubj', 'ROOT']:
                                    potential_subjects.append(token.text)
                            
                            if potential_subjects:
                                subject = potential_subjects[0]
                                # Add the subject to the beginning of the sentence
                                if text[0].islower():
                                    text = f"{subject} {text}"
                                else:
                                    text = f"{subject} {text[0].lower() + text[1:]}"
                
                # 5. Remove double periods
                text = text.replace('..', '.')
                
                # 6. Capitalize first letter of sentences
                if text and text[0].islower():
                    text = text[0].upper() + text[1:]
                
                # Update the natural text with fixed version
                block.natural_text = text
            
        return blocks

    def group_into_scenes(self, blocks: List[ContentBlock]) -> List[Scene]:
        """Group content blocks into scenes based on proper heading hierarchy."""
        # First, identify all potential scene boundaries (real headings)
        scene_boundaries = []
        current_heading_level = 0
        
        for i, block in enumerate(blocks):
            # Only consider real headings/titles, not bullet points
            if (block.content_type in [ContentType.TITLE, ContentType.HEADING] and 
                not block.content.startswith('•') and 
                not block.content.startswith('-')):
                
                # Get the heading level from the content (# = 1, ## = 2, etc.)
                heading_level = 1
                if block.content.startswith('#'):
                    heading_level = len(re.match(r'^#+', block.content).group(0))
                
                # Consider this a scene boundary if:
                # 1. It's the first heading we've seen
                # 2. It's a level 1 or 2 heading
                # 3. It's at the same or higher level as the current heading level
                if len(scene_boundaries) == 0 or heading_level <= 2 or heading_level <= current_heading_level:
                    scene_boundaries.append(i)
                    current_heading_level = heading_level
        
        # If no scene boundaries found, treat the entire script as one scene
        if not scene_boundaries:
            scene_boundaries = [0]
        
        # Now group blocks into scenes based on boundaries
        scenes = []
        for scene_idx, boundary_idx in enumerate(scene_boundaries):
            # Determine the end of this scene
            next_boundary_idx = scene_boundaries[scene_idx + 1] if scene_idx + 1 < len(scene_boundaries) else len(blocks)
            
            # Get the blocks for this scene
            scene_blocks = blocks[boundary_idx:next_boundary_idx]
            
            # Skip empty scenes
            if not scene_blocks:
                continue
                
            # Determine scene title (use first block if it's a heading, otherwise use default)
            title_block = scene_blocks[0]
            scene_title = title_block.content if title_block.content_type in [ContentType.TITLE, ContentType.HEADING] else "Scene"
            
            # Extract metadata for the scene
            scene_type = self._determine_scene_type(scene_blocks, scene_title)
            states, weighted_features = self._extract_states_and_features(scene_blocks)
            visual_cues = self._extract_visual_cues(scene_blocks)
            
            # Extract feature texts for backward compatibility
            feature_texts = [feature["text"] for feature in weighted_features]
            
            # Create natural content by joining all natural texts
            natural_content = " ".join([
                block.natural_text or block.content 
                for block in scene_blocks
            ])
            
            # Calculate complexity score with metrics
            complexity_score, complexity_metadata = self._calculate_complexity(natural_content)
            
            # Create scene
            scene = Scene(
                id=scene_idx + 1,
                title=scene_title,
                blocks=scene_blocks,
                scene_type=scene_type,
                states=states,
                features=feature_texts,  # Keep backward compatibility
                visual_cues=visual_cues,
                natural_content=natural_content,
                errors=[],
                metadata={
                    "word_count": len(natural_content.split()),
                    "estimated_duration": len(natural_content.split()) * 0.5,  # 0.5 seconds per word
                    "weighted_features": weighted_features,  # Add the weighted features
                    "complexity_score": complexity_score,
                    "complexity_metrics": complexity_metadata  # Ensure complexity metrics are stored
                }
            )
            
            scenes.append(scene)
        
        return scenes

    def _calculate_complexity(self, text: str) -> Tuple[float, Dict]:
        """
        Calculate the complexity score of the text using advanced NLP metrics.
        Returns a tuple (complexity score, complexity metadata).
        """
        if not text.strip():
            return 0.0, {}
            
        doc = self.nlp(text)
        
        # Basic counts
        word_count = len([token for token in doc if not token.is_punct and not token.is_space])
        if word_count == 0:
            return 0.0, {}
            
        # 1. Lexical Complexity Measures
        # - Average word length
        avg_word_length = sum(len(token.text) for token in doc if not token.is_punct and not token.is_space) / word_count if word_count > 0 else 0
        
        # - Percentage of complex words (long words or low-frequency)
        complex_words = sum(1 for token in doc if len(token.text) > 7 and not token.is_punct and not token.is_space)
        complex_word_ratio = complex_words / word_count if word_count > 0 else 0
        
        # - Lexical diversity (unique words / total words)
        unique_words = len(set([token.text.lower() for token in doc if not token.is_punct and not token.is_space]))
        lexical_diversity = unique_words / word_count if word_count > 0 else 0
        
        # 2. Syntactic Complexity Measures
        sentences = list(doc.sents)
        sentence_count = len(sentences)
        
        # - Average sentence length (words per sentence)
        avg_sentence_length = word_count / sentence_count if sentence_count > 0 else 0
        
        # - Clausal density (number of clauses per sentence)
        # Simple approximation: count verbs as potential clauses
        verbs_per_sentence = sum(1 for token in doc if token.pos_ == "VERB") / sentence_count if sentence_count > 0 else 0
        
        # - Subordination (look for subordinating conjunctions or relative pronouns)
        subordinators = ["because", "although", "though", "since", "while", "whereas", 
                         "if", "unless", "until", "when", "whenever", "where", "wherever", 
                         "which", "who", "whom", "whose", "that"]
        
        subordination_count = sum(1 for token in doc 
                                 if token.text.lower() in subordinators 
                                 or token.dep_ in ["mark", "relcl"])
        
        subordination_ratio = subordination_count / sentence_count if sentence_count > 0 else 0
        
        # 3. Semantic Complexity
        # - Named entity density (more named entities suggest more specific, complex content)
        entities = len(doc.ents)
        entity_density = entities / word_count if word_count > 0 else 0
        
        # - Academic/specialized vocabulary
        academic_pos = ["ADJ", "NOUN", "VERB"]  # Parts of speech common in academic texts
        academic_words = sum(1 for token in doc 
                             if token.pos_ in academic_pos 
                             and len(token.text) > 5 
                             and not token.is_stop)
        
        academic_ratio = academic_words / word_count if word_count > 0 else 0
        
        # NEW: 4. Readability metrics using textstat
        try:
            # Calculate various readability scores
            flesch_reading_ease = textstat.flesch_reading_ease(text)
            flesch_kincaid_grade = textstat.flesch_kincaid_grade(text)
            gunning_fog = textstat.gunning_fog(text)
            smog_index = textstat.smog_index(text)
            
            # Normalize these scores to 0-1 range for our complexity measure
            # For Flesch Reading Ease, higher means easier, so we invert
            normalized_flesch = 1 - (max(0, min(flesch_reading_ease, 100)) / 100)
            
            # For the others, higher means more complex
            # Most grade-level scores go up to around 18 (graduate level)
            normalized_fk_grade = min(flesch_kincaid_grade, 18) / 18
            normalized_gunning_fog = min(gunning_fog, 18) / 18
            normalized_smog = min(smog_index, 18) / 18
            
            # Combine readability metrics
            readability_complexity = (
                normalized_flesch + 
                normalized_fk_grade + 
                normalized_gunning_fog + 
                normalized_smog
            ) / 4
            
        except Exception as e:
            # If textstat calculation fails, use a fallback
            print(f"Warning: Readability metrics calculation failed: {str(e)}")
            readability_complexity = (avg_sentence_length / 30) * 0.5 + complex_word_ratio * 0.5

        # NEW: 5. Grammar complexity - analyze sentence structures
        # - Count sentence types
        sentence_types = {
            "simple": 0,
            "compound": 0,
            "complex": 0,
            "compound_complex": 0
        }
        
        for sent in sentences:
            # Simple approximation method:
            # - Count coordinating conjunctions (and, but, or) for compound sentences
            # - Count subordinating conjunctions for complex sentences
            # - If both are present, it's compound-complex
            
            has_coord = any(token.text.lower() in ["and", "but", "or", "so", "yet"] and token.dep_ == "cc" 
                           for token in sent)
            
            has_subord = any(token.text.lower() in subordinators or token.dep_ in ["mark", "relcl"] 
                            for token in sent)
            
            if has_coord and has_subord:
                sentence_types["compound_complex"] += 1
            elif has_coord:
                sentence_types["compound"] += 1
            elif has_subord:
                sentence_types["complex"] += 1
            else:
                sentence_types["simple"] += 1
                
        # Calculate weighted grammar complexity
        if sentence_count > 0:
            # Assign weights to different sentence types (higher for more complex types)
            weighted_sentence_score = (
                sentence_types["simple"] * 0.25 +
                sentence_types["compound"] * 0.5 +
                sentence_types["complex"] * 0.75 +
                sentence_types["compound_complex"] * 1.0
            ) / sentence_count
        else:
            weighted_sentence_score = 0
        
        # 6. Combine all factors into a weighted complexity score with the new metrics
        lexical_complexity = 0.25 * (
            (avg_word_length / 10) + 
            complex_word_ratio + 
            lexical_diversity
        ) / 3
        
        syntactic_complexity = 0.25 * (
            (avg_sentence_length / 30) + 
            (verbs_per_sentence / 5) + 
            subordination_ratio
        ) / 3
        
        semantic_complexity = 0.2 * (
            entity_density +
            academic_ratio
        ) / 2
        
        grammar_complexity = 0.15 * weighted_sentence_score
        
        readability_factor = 0.15 * readability_complexity
        
        # Combine all types of complexity
        total_complexity = (
            lexical_complexity + 
            syntactic_complexity + 
            semantic_complexity + 
            grammar_complexity + 
            readability_factor
        )
        
        # Normalize to 0-1 range
        normalized_complexity = min(max(total_complexity, 0.0), 1.0)
        
        # Store detailed metrics in metadata
        complexity_metadata = {
            "lexical_complexity": round(lexical_complexity, 3),
            "syntactic_complexity": round(syntactic_complexity, 3),
            "semantic_complexity": round(semantic_complexity, 3),
            "grammar_complexity": round(grammar_complexity, 3),
            "readability_complexity": round(readability_factor, 3),
            "avg_word_length": round(avg_word_length, 2),
            "complex_word_ratio": round(complex_word_ratio, 3),
            "lexical_diversity": round(lexical_diversity, 3),
            "avg_sentence_length": round(avg_sentence_length, 2),
            "clausal_density": round(verbs_per_sentence, 2),
            "subordination_ratio": round(subordination_ratio, 3),
            "entity_density": round(entity_density, 3),
            "academic_ratio": round(academic_ratio, 3),
            "sentence_types": {
                "simple": sentence_types["simple"],
                "compound": sentence_types["compound"],
                "complex": sentence_types["complex"],
                "compound_complex": sentence_types["compound_complex"]
            }
        }
        
        # Add readability metrics if available
        try:
            complexity_metadata.update({
                "flesch_reading_ease": round(flesch_reading_ease, 2),
                "flesch_kincaid_grade": round(flesch_kincaid_grade, 2),
                "gunning_fog": round(gunning_fog, 2),
                "smog_index": round(smog_index, 2)
            })
        except:
            pass
        
        return normalized_complexity, complexity_metadata

    def generate_structured_output(self, scenes: List[Scene]) -> Dict:
        """Generate structured output in the required format."""
        scene_data = []
        total_words = 0
        total_complexity = 0
        total_duration = 0
        
        # Count diagram and image requirements
        diagram_count = 0
        image_count = 0
        
        for scene in scenes:
            # Update counts
            total_words += scene.metadata["word_count"]
            complexity_score = scene.metadata["complexity_score"]
            if isinstance(complexity_score, tuple):
                # If complexity is a tuple (score, metadata), extract just the score
                complexity_score = complexity_score[0]
            total_complexity += complexity_score
            total_duration += scene.metadata["estimated_duration"]
            
            # Check for visual requirements
            if scene.scene_type == SceneType.DIAGRAM_REQUIRED:
                diagram_count += 1
            elif scene.scene_type == SceneType.IMAGE_REQUIRED:
                image_count += 1
                
            # Count explicit visual cues
            for cue in scene.visual_cues:
                if "diagram" in cue.lower():
                    diagram_count += 1
                elif any(img_word in cue.lower() for img_word in ["image", "picture", "photo"]):
                    image_count += 1
            
            # Add weighted features to the output
            weighted_features = scene.metadata.get("weighted_features", [])
            
            # Create scene data
            scene_data_entry = {
                "scene": scene.id,
                "title": scene.title,
                "scene_type": scene.scene_type.value,
                "states": scene.states,
                "features": scene.features,
                "weighted_features": weighted_features,  # New field with importance scores
                "visual_cues": scene.visual_cues,
                "natural_content": scene.natural_content,
                "metadata": {
                    "word_count": scene.metadata["word_count"],
                    "estimated_duration": scene.metadata["estimated_duration"]
                }
            }
            
            # Handle complexity score and metrics
            if "complexity_score" in scene.metadata:
                scene_data_entry["metadata"]["complexity_score"] = scene.metadata["complexity_score"]
                
                # Make sure complexity metrics are also included when available
                if "complexity_metrics" in scene.metadata:
                    scene_data_entry["metadata"]["complexity_metrics"] = scene.metadata["complexity_metrics"]
            
            # Add additional metadata
            if "pacing_weight" in scene.metadata:
                scene_data_entry["metadata"]["pacing_weight"] = scene.metadata["pacing_weight"]
                
            scene_data.append(scene_data_entry)
        
        # Calculate average complexity (handling the case where complexity might be a tuple)
        if scenes:
            avg_complexity = total_complexity / len(scenes)
        else:
            avg_complexity = 0
            
        # Build final output
        output = {
            "metadata": {
                "total_scenes": len(scenes),
                "total_words": total_words,
                "average_complexity": avg_complexity,
                "estimated_duration": total_duration,
                "visual_requirements": {
                    "total_diagrams": diagram_count,
                    "total_images": image_count
                }
            },
            "scenes": scene_data
        }
        
        return output

    def parse_script(self, script: str, file_path: str = None) -> Dict:
        """
        Parse a script and generate structured output.
        
        Args:
            script: The script text to parse
            file_path: Optional path to the script file (used for PDF extraction)
            
        Returns:
            A dictionary with the structured output of the parsed script
        """
        try:
            # Step 1: Handle file-based input if file_path is provided
            if file_path and not script:
                script = self.extract_text_from_file(file_path)
            
            # Step 2: Preprocess the script
            preprocessed_script = self._preprocess_script(script)
            
            # Step 3: Segment content into blocks
            blocks = self.segment_content(preprocessed_script)
            
            # Step 4: Detect and fix errors in content
            blocks = self._detect_and_fix_errors(blocks)
            
            # Step 5: Group into scenes
            scenes = self.group_into_scenes(blocks)
            
            # Step 6: Generate structured output
            result = self.generate_structured_output(scenes)
            
            # Step 7: Add parser version and timestamp
            result["parser_metadata"] = {
                "version": "1.2.0",
                "timestamp": self._get_timestamp(),
                "enhancements": [
                    "visual_cue_markers",
                    "feature_importance_weighting",
                    "advanced_complexity_metrics",
                    "grammar_analysis",
                    "readability_metrics"
                ]
            }
            
            return result
            
        except Exception as e:
            print(f"Error parsing script: {str(e)}")
            import traceback
            traceback.print_exc()
            raise
            
    def _get_timestamp(self) -> str:
        """Get the current timestamp in ISO format."""
        from datetime import datetime
        return datetime.now().isoformat() 