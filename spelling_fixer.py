# Adrienne Dominique Sy Chu
# Description: Hidden Markov Model spell fixer using Viterbi.

import math
import os
from collections import defaultdict, Counter

# Use this instead of -infinity to avoid math errors
LOG_ZERO = -1e9

# Return log(x), handling zeros safely.
def safe_log(x):
    if x <= 0:
        return LOG_ZERO
    return math.log(x)

# Read aspell.txt and calculate emission and transition probabilities.
# Returns dict with emission (E), transition (T), and start probabilities.
def build_hmm_from_aspell(filename):
    alphabet = list("abcdefghijklmnopqrstuvwxyz'")
    
    # Counters for building probabilities
    emission_counts = defaultdict(Counter)  # emission_counts[correct][typed]
    transition_counts = defaultdict(Counter)  # transition_counts[curr][next]
    start_counts = Counter()  # start_counts[letter]
    
    # Check if file exists
    if not os.path.exists(filename):
        print(f"Error: '{filename}' not found!")
        return None
    
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip().lower()
            if not line or ':' not in line:
                continue
            
            # Split: "correct: typo1 typo2 typo3"
            parts = line.split(':', 1)
            correct_word = parts[0].strip()
            typos = parts[1].strip().split()
            
            # Filter correct word to alphabet only
            correct_letters = [c for c in correct_word if c in alphabet]
            if len(correct_letters) == 0:
                continue
            
            # Learn emissions from SAME-LENGTH typos only
            # (This is the key limitation of character-level HMM)
            for typo in typos:
                typo_letters = [c for c in typo if c in alphabet]
                
                # Only align if same length
                if len(correct_letters) == len(typo_letters):
                    for i in range(len(correct_letters)):
                        correct_char = correct_letters[i]
                        typed_char = typo_letters[i]
                        # Count: when correct letter is X, typed letter is Y
                        emission_counts[correct_char][typed_char] += 1
            
            # Learn transitions from correct word only
            start_counts[correct_letters[0]] += 1
            for i in range(len(correct_letters) - 1):
                curr = correct_letters[i]
                next_char = correct_letters[i + 1]
                transition_counts[curr][next_char] += 1
    
    # Convert counts to probabilities with smoothing
    V = len(alphabet)
    smoothing = 0.01
    
    # Emission probabilities: P(typed | correct)
    E = {}
    for correct in alphabet:
        E[correct] = {}
        # Add smoothing to avoid zero probabilities
        total = sum(emission_counts[correct].values()) + V * smoothing
        for typed in alphabet:
            count = emission_counts[correct][typed]
            # Boost diagonal, but not too much!
            # We want transitions to be able to override when needed
            if correct == typed:
                count += 5  # moderate boost for correct typing
            E[correct][typed] = (count + smoothing) / total
    
    # Transition probabilities: P(next | current)
    T = {}
    for curr in alphabet:
        T[curr] = {}
        total = sum(transition_counts[curr].values()) + V * smoothing
        for next_char in alphabet:
            count = transition_counts[curr][next_char]
            T[curr][next_char] = (count + smoothing) / total
    
    # Start probabilities: P(first letter)
    start_prob = {}
    total_start = sum(start_counts.values()) + V * smoothing
    for letter in alphabet:
        count = start_counts[letter]
        start_prob[letter] = (count + smoothing) / total_start
    
    return {
        'alphabet': alphabet,
        'E': E,
        'T': T,
        'start': start_prob,
        'emission_counts': emission_counts  # return raw counts for debugging
    }

# Viterbi algorithm: find most likely sequence of correct letters.
def viterbi(observations, hmm):
    alphabet = hmm['alphabet']
    E = hmm['E']  # emission probabilities
    T = hmm['T']  # transition probabilities
    start = hmm['start']  # start probabilities
    
    # Filter to alphabet only
    O = [c for c in observations.lower() if c in alphabet]
    if len(O) == 0:
        return observations
    
    t = len(O)  # number of time steps
    
    # M[time][state] = max log probability of being in state at time
    # Backpointers[time][state] = best previous state
    M = [{} for _ in range(t)]
    Backpointers = [{} for _ in range(t)]
    
    # INITIALIZATION: time step 0
    # M[0][s] = P(start with s) * P(observe O[0] | state s)
    for state in alphabet:
        M[0][state] = safe_log(start[state]) + safe_log(E[state][O[0]])
        Backpointers[0][state] = None
    
    # RECURSION: time steps 1 to t-1
    for time in range(1, t):
        for state in alphabet:
            max_val = LOG_ZERO
            best_prev = None
            
            # Try all possible previous states
            for prev_state in alphabet:
                # Calculate: prev_prob + transition + emission
                val = (M[time-1][prev_state] + 
                        safe_log(T[prev_state][state]) + 
                        safe_log(E[state][O[time]]))
                
                if val > max_val:
                    max_val = val
                    best_prev = prev_state
            
            M[time][state] = max_val
            Backpointers[time][state] = best_prev
    
    # TERMINATION: find best final state
    best = max(M[t-1], key=M[t-1].get)
    
    # BACKTRACK: reconstruct path
    path = []
    for time in range(t-1, -1, -1):
        path.append(best)
        if time > 0:
            best = Backpointers[time][best]
    
    path.reverse()
    return ''.join(path)

# Split text into words and correct each word.
def correct_sentence(text, hmm):
    words = text.split()
    corrected = []
    for word in words:
        corrected_word = viterbi(word, hmm)
        corrected.append(corrected_word)
    return ' '.join(corrected)

def main():
    # Build HMM from training data
    print("Building HMM from aspell.txt...")
    hmm = build_hmm_from_aspell('aspell.txt')
    
    if hmm is None:
        return
    
    print("Done!\n")
    
    # Show some statistics
    print("=" * 60)
    print("HMM Statistics")
    print("=" * 60)
    
    # DEBUG: Check what we learned from the training data
    emission_counts = hmm['emission_counts']
    print("\nDEBUG: Raw emission counts (before adding boost):")
    print("  'i' -> 'e' count:", emission_counts['i']['e'])
    print("  'i' -> 'i' count:", emission_counts['i']['i'])
    print("  'e' -> 'i' count:", emission_counts['e']['i'])
    print("  'e' -> 'e' count:", emission_counts['e']['e'])
    print("  'h' -> 't' count:", emission_counts['h']['t'])
    print("  'e' -> 'a' count:", emission_counts['e']['a'])
    
    # Show emission probabilities
    print("\nEmission probabilities for 'i' (top 5):")
    i_probs = sorted(hmm['E']['i'].items(), key=lambda x: -x[1])[:5]
    for typed, prob in i_probs:
        print(f"  P(typed='{typed}' | correct='i') = {prob:.4f}")
    
    print("\nEmission probabilities for 'e' (top 5):")
    e_probs = sorted(hmm['E']['e'].items(), key=lambda x: -x[1])[:5]
    for typed, prob in e_probs:
        print(f"  P(typed='{typed}' | correct='e') = {prob:.4f}")
    
    # Show key emission probabilities for believe/beleive
    print("\nKey probabilities for 'believe' vs 'beleive':")
    print(f"  P(typed='e' | correct='i') = {hmm['E']['i']['e']:.6f}")
    print(f"  P(typed='i' | correct='i') = {hmm['E']['i']['i']:.6f}")
    print(f"  P(typed='i' | correct='e') = {hmm['E']['e']['i']:.6f}")
    print(f"  P(typed='e' | correct='e') = {hmm['E']['e']['e']:.6f}")
    
    # Show transition probabilities
    print("\nTransition probabilities from 'e' (top 5):")
    e_trans = sorted(hmm['T']['e'].items(), key=lambda x: -x[1])[:5]
    for next_letter, prob in e_trans:
        print(f"  P('{next_letter}' | 'e') = {prob:.4f}")
    
    # Show start probabilities
    print("\nStart probabilities (top 5):")
    starts = sorted(hmm['start'].items(), key=lambda x: -x[1])[:5]
    for letter, prob in starts:
        print(f"  P(start with '{letter}') = {prob:.4f}")
    
    # Interactive loop
    print("\n" + "=" * 60)
    print("Spelling Corrector")
    print("Type text to correct (blank line to quit)")
    print("=" * 60)
    
    while True:
        text = input("\nEnter text: ").strip()
        if not text:
            print("Goodbye!")
            break
        
        corrected = correct_sentence(text, hmm)
        print(f"Corrected: {corrected}")

if __name__ == '__main__':
    main()