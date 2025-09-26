import heapq
import os

# Represents a state during alignment search between two documents
class AlignmentState:
    def __init__(self, i, j, cost, path):
        self.i = i  # Current index in first document
        self.j = j  # Current index in second document
        self.g = cost  # Cost accumulated so far
        self.path = path  # List of alignment actions performed
        self.f = 0  # f(n) = g(n) + h(n), heuristic added externally

    def __lt__(self, other):
        return self.f < other.f

# Compute Levenshtein edit distance between two strings s and t
def edit_dist(s, t):
    if len(s) < len(t):
        return edit_dist(t, s)
    if len(t) == 0:
        return len(s)
    prev = list(range(len(t) + 1))
    for i, c1 in enumerate(s):
        curr = [i + 1]
        for j, c2 in enumerate(t):
            ins = prev[j + 1] + 1
            del_cost = curr[j] + 1
            sub = prev[j] + (c1 != c2)
            curr.append(min(ins, del_cost, sub))
        prev = curr
    return prev[-1]

# Simple heuristic estimating remaining alignment cost based on sentence counts left
def h(state, n1, n2):
    rem1 = n1 - state.i
    rem2 = n2 - state.j
    return abs(rem1 - rem2)

# Given a state, generate all next possible states with costs
def get_next(state, doc1, doc2):
    next_states = []
    i, j = state.i, state.j

    # Align sentences if both indices are valid
    if i < len(doc1) and j < len(doc2):
        cost = edit_dist(doc1[i], doc2[j])
        new_path = state.path + [("match", i, j, cost)]
        next_states.append(AlignmentState(i + 1, j + 1, state.g + cost, new_path))

    # Skip sentence in first document
    if i < len(doc1):
        cost = len(doc1[i])
        new_path = state.path + [("skip1", i, None, cost)]
        next_states.append(AlignmentState(i + 1, j, state.g + cost, new_path))

    # Skip sentence in second document
    if j < len(doc2):
        cost = len(doc2[j])
        new_path = state.path + [("skip2", None, j, cost)]
        next_states.append(AlignmentState(i, j + 1, state.g + cost, new_path))

    return next_states

# Detect plagiarism between two documents by A* alignment and similarity thresholding
def find_plagiarism(doc1, doc2, thresh=0.5):
    start = AlignmentState(0, 0, 0, [])
    n1, n2 = len(doc1), len(doc2)
    q = []
    heapq.heappush(q, (start.g, start))
    visited = set()

    while q:
        _, curr = heapq.heappop(q)
        key = (curr.i, curr.j)
        if key in visited:
            continue
        visited.add(key)

        if curr.i == n1 and curr.j == n2:
            matches = []
            all_alignments = []
            for step in curr.path:
                op, x, y, cost = step
                if op == "match":
                    s1, s2 = doc1[x], doc2[y]
                    max_len = max(len(s1), len(s2))
                    sim = 1 - cost / max_len if max_len > 0 else 0
                    all_alignments.append(sim)
                    if sim >= thresh:
                        matches.append((s1, s2, sim))
            
            # Calculate score as average similarity of all sentence pairs
            avg_similarity = sum(all_alignments) / len(all_alignments) if all_alignments else 0
            high_similarity_ratio = len(matches) / len(all_alignments) if all_alignments else 0
            
            return {
                "avg_score": avg_similarity,
                "high_sim_ratio": high_similarity_ratio,
                "matches": matches,
                "all_sims": all_alignments,
                "path": curr.path,
                "cost": curr.g
            }

        for next_state in get_next(curr, doc1, doc2):
            next_state.f = next_state.g + h(next_state, n1, n2)
            if (next_state.i, next_state.j) not in visited:
                heapq.heappush(q, (next_state.f, next_state))

    return None

# A class to wrap preprocessing and detection functionalities
class Detector:
    def __init__(self, thresh=0.5):
        self.thresh = thresh

    def prep(self, txt):
        sents = [s.strip() for s in txt.split('.') if s.strip()]
        return [s.lower() for s in sents]

    def load_file(self, filename):
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                return f.read()
        except FileNotFoundError:
            print(f"File {filename} not found!")
            return ""

    def check(self, txt1, txt2):
        d1 = self.prep(txt1)
        d2 = self.prep(txt2)

        res = find_plagiarism(d1, d2, self.thresh)
        if res:
            return {
                "plagiarism_detected": res["avg_score"] > 0.6,
                "avg_similarity": res["avg_score"],
                "high_similarity_matches": len(res["matches"]),
                "matches": res["matches"],
                "all_similarities": res["all_sims"]
            }
        else:
            return {"plagiarism_detected": False, "avg_similarity": 0, "high_similarity_matches": 0}

# Run tests with files
def test_files():
    detector = Detector(0.5)
    
    base_doc = detector.load_file("doc1.txt")
    files = ["doc2.txt", "doc3.txt", "doc4.txt"]

    print(" Starting Plagiarism Detection System\n")
    print(" Using A* Search Algorithm for Text Alignment\n")
    
    for file in files:
        test_doc = detector.load_file(file)
        
        if base_doc and test_doc:
            res = detector.check(base_doc, test_doc)
            
            print(f"Comparing: doc1.txt with {file}")
  
            print(f"Average Similarity Score: {res['avg_similarity']:.3f}")
            print(f"High Similarity Matches: {res['high_similarity_matches']}")
            print(f"Plagiarism Detected: {'YES' if res['plagiarism_detected'] else 'NO'}")
            
            if res.get("matches"):
                print(f"\nSimilar Sentence Pairs Found:")
                for k, (s1, s2, sim) in enumerate(res["matches"][:5], 1):
                    print(f"  {k}. Similarity: {sim:.3f}")
                    print(f"     doc1.txt: '{s1[:60]}{'...' if len(s1) > 60 else ''}'")
                    print(f"     {file}: '{s2[:60]}{'...' if len(s2) > 60 else ''}'")
                    print()
                    
                if len(res["matches"]) > 5:
                    print(f" and {len(res['matches']) - 5} more matches\n")
            else:
                print("\n No significant similarities found")
                
            if res.get("all_similarities"):
                avg_all = sum(res["all_similarities"]) / len(res["all_similarities"])
                


if __name__ == "__main__":
    test_files()
