# ===================================================================
# File: run_corrected_evaluation.py  
# Runner script for corrected RL vs IL evaluation
# ===================================================================

import sys
import os
from pathlib import Path

def main():
    """
    Run the corrected evaluation that uses actual trained models
    """
    
    print("🚀 CORRECTED RL vs IL EVALUATION")
    print("=" * 50)
    print("✅ Using actual trained models (no pseudo simulation)")
    print("✅ Fair comparison with same world model as simulator")
    print("✅ Rigorous statistical analysis")
    print("✅ Complete LaTeX paper generation")
    print("=" * 50)
    
    # Import the corrected evaluation framework
    try:
        from corrected_rl_vs_il_evaluation import run_corrected_evaluation
        
        # Run the evaluation
        evaluator, results = run_corrected_evaluation()
        
        if evaluator and results:
            print("\n🎯 SUCCESS! Your corrected evaluation is complete.")
            print("\nNext steps:")
            print("1. Review the generated paper: corrected_publication_results/complete_paper.tex")
            print("2. Check the main figure: corrected_publication_results/comprehensive_evaluation_results.pdf")
            print("3. Use the LaTeX tables: corrected_publication_results/publication_tables.tex")
            print("4. Examine detailed results: corrected_publication_results/comprehensive_results.json")
        else:
            print("\n❌ Evaluation failed - check error messages above")
            
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure corrected_rl_vs_il_evaluation.py is in the same directory")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
