from src.script_parser.core import ScriptParser
import json

def main():
    # Initialize the parser
    parser = ScriptParser()
    
    # Get the script file to test (default or enhanced)
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "enhanced":
        script_file = "test_enhanced_script.txt"
        print("\n=== 📄 Processing ENHANCED script... ===\n")
    else:
        script_file = "test_script.txt"
        print("\n=== 📄 Processing standard script... ===\n")
    
    # Read the test script
    with open(script_file, "r", encoding="utf-8") as f:
        script = f.read()
    
    # Parse the script
    result = parser.parse_script(script)
    
    # Save the JSON output filename based on input
    output_file = "parsed_enhanced_script.json" if script_file == "test_enhanced_script.txt" else "parsed_script.json"
    
    # Print the results
    print("\n=== 🔍 Script Analysis Results ===\n")
    
    # Print metadata
    print("📊 Metadata:")
    print(f"  Total Scenes: {result['metadata']['total_scenes']}")
    print(f"  Total Words: {result['metadata']['total_words']}")
    print(f"  Estimated Duration: {result['metadata']['estimated_duration']:.2f} seconds")
    print(f"  Average Complexity: {result['metadata']['average_complexity']:.2f}")
    
    print("\n🖼️ Visual Requirements:")
    print(f"  Diagrams Needed: {result['metadata']['visual_requirements']['total_diagrams']}")
    print(f"  Images Needed: {result['metadata']['visual_requirements']['total_images']}")
    
    # Print scene details
    print("\n📝 Scene Analysis:")
    for scene in result["scenes"]:
        print(f"\n✨ Scene {scene['scene']}: {scene['title']}")
        print(f"  Type: {scene['scene_type']}")
        
        if scene['states']:
            print(f"  States: {', '.join(scene['states'])}")
        
        if scene.get('weighted_features'):
            print("\n  📊 Weighted Features (by importance):")
            for feature in scene['weighted_features']:
                importance_stars = "★" * int(feature["importance"] * 5 + 0.5)  # Round to nearest star (max 5)
                print(f"    • {importance_stars} ({feature['importance']:.2f}) {feature['text']}")
        elif scene['features']:
            print("\n  Features (Bullet Points):")
            for feature in scene['features']:
                print(f"    • {feature}")
        
        # Display the natural content
        if scene['natural_content']:
            print("\n  📚 Natural Content:")
            # Split into lines for better readability
            natural_content_lines = scene['natural_content'].split('. ')
            for line in natural_content_lines:
                if line.strip():
                    print(f"    • {line.strip()}.")
        
        if scene['visual_cues']:
            print("\n  🖌️ Visual Cues:")
            for cue in scene['visual_cues']:
                print(f"    • {cue}")
        
        # Display complexity metrics if available
        if 'complexity_metrics' in scene['metadata']:
            print("\n  📏 Advanced Complexity Metrics:")
            metrics = scene['metadata']['complexity_metrics']
            print(f"    • Lexical Complexity: {metrics['lexical_complexity']:.3f}")
            print(f"    • Syntactic Complexity: {metrics['syntactic_complexity']:.3f}")
            print(f"    • Semantic Complexity: {metrics['semantic_complexity']:.3f}")
            print(f"    • Average Word Length: {metrics['avg_word_length']:.2f}")
            print(f"    • Lexical Diversity: {metrics['lexical_diversity']:.3f}")
            print(f"    • Average Sentence Length: {metrics['avg_sentence_length']:.2f} words")
        
        print(f"\n  Complexity Score: {scene['metadata']['complexity_score']:.2f}")
        print(f"  Estimated Duration: {scene['metadata']['estimated_duration']:.2f} seconds")
        print("-" * 70)
    
    # Save the structured output to a JSON file
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    
    print(f"\n✅ Structured output saved to {output_file}\n")
    
    # Validate the results
    print("\n=== 🧪 Validation Report ===\n")
    
    # Check for hallucinations
    hallucination_keywords = ["CNN", "dailymail", "www.", "http", ".com"]
    has_hallucinations = False
    
    for scene in result["scenes"]:
        for keyword in hallucination_keywords:
            if keyword in scene['natural_content'] and not keyword in script:
                has_hallucinations = True
                print(f"❌ Hallucination detected in Scene {scene['scene']}: '{keyword}'")
    
    if not has_hallucinations:
        print("✅ No hallucinations detected in natural content")
    
    # Check if features are populated
    if any('weighted_features' in scene for scene in result['scenes']):
        print("✅ Enhanced feature weighting is implemented")
    
    # Check if visual cues are detected from @@markers
    has_marker_detection = False
    for scene in result['scenes']:
        for cue in scene.get('visual_cues', []):
            if '@@' in script and ('Diagram:' in cue or 'Image:' in cue):
                has_marker_detection = True
                break
    
    if has_marker_detection or '@@' not in script:
        print("✅ Visual cue marker detection is working")
    else:
        print("❌ Visual cue marker detection failed")
    
    # Check for advanced complexity metrics
    has_advanced_complexity = any('complexity_metrics' in scene['metadata'] for scene in result['scenes'])
    if has_advanced_complexity:
        print("✅ Advanced complexity metrics are implemented")
    else:
        print("❌ Advanced complexity metrics are missing")
    
    # Check for sensible scene count
    if 2 <= len(result["scenes"]) <= 10:
        print(f"✅ Scene count is reasonable: {len(result['scenes'])}")
    else:
        print(f"❌ Unexpected scene count: {len(result['scenes'])}")
    
    # Check that bullet points aren't treated as scene titles
    bullet_titles = [scene['title'] for scene in result["scenes"] if scene['title'].startswith('•')]
    if not bullet_titles:
        print("✅ No bullet points detected as scene titles")
    else:
        print(f"❌ Bullet points incorrectly used as scene titles: {len(bullet_titles)}")
    
    # Check parser version
    if 'parser_metadata' in result and 'version' in result['parser_metadata']:
        print(f"✅ Parser version: {result['parser_metadata']['version']}")
        
        if 'enhancements' in result['parser_metadata']:
            print(f"✅ Enhancements: {', '.join(result['parser_metadata']['enhancements'])}")

if __name__ == "__main__":
    main() 