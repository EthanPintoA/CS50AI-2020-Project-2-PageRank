import os
import random
import re
import sys

DAMPING = 0.85
SAMPLES = 10000


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python pagerank.py corpus")
    corpus = crawl(sys.argv[1])
    ranks = sample_pagerank(corpus, DAMPING, SAMPLES)
    print(f"PageRank Results from Sampling (n = {SAMPLES})")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")
    ranks = iterate_pagerank(corpus, DAMPING)
    print(f"PageRank Results from Iteration")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")


def crawl(directory):
    """
    Parse a directory of HTML pages and check for links to other pages.
    Return a dictionary where each key is a page, and values are
    a list of all other pages in the corpus that are linked to by the page.
    """
    pages = dict()

    # Extract all links from HTML files
    for filename in os.listdir(directory):
        if not filename.endswith(".html"):
            continue
        with open(os.path.join(directory, filename)) as f:
            contents = f.read()
            links = re.findall(r"<a\s+(?:[^>]*?)href=\"([^\"]*)\"", contents)
            pages[filename] = set(links) - {filename}

    # Only include links to other pages in the corpus
    for filename in pages:
        pages[filename] = set(
            link for link in pages[filename]
            if link in pages
        )

    return pages


def transition_model(corpus, page, damping_factor):
    """
    Return a probability distribution over which page to visit next,
    given a current page.

    With probability `damping_factor`, choose a link at random
    linked to by `page`. With probability `1 - damping_factor`, choose
    a link at random chosen from all pages in the corpus.
    """
    if len(corpus[page]):
        model = {}

        for link in corpus:
            model[link] = (1.0 - damping_factor) / len(corpus)

            if link in corpus[page]:
                model[link] += damping_factor / len(corpus[page])

        return model
    else:
        # If `page` has no links,
        # we can pretend it has links to all pages in `corpus`
        return {link: 1.0 / len(corpus) for link in corpus}


def sample_pagerank(corpus, damping_factor, n):
    """
    Return PageRank values for each page by sampling `n` pages
    according to transition model, starting with a page at random.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    ranks = {page: 0.0 for page in corpus}
    sample = random.choice(tuple(corpus))

    for _ in range(n):
        ranks[sample] += 1.0 / n

        model = transition_model(corpus, sample, damping_factor)

        sample = random.choices(tuple(model), model.values())[0]

    return ranks


def iterate_pagerank(corpus, damping_factor):
    """
    Return PageRank values for each page by iteratively updating
    PageRank values until convergence.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    ranks = {page: 1.0 / len(corpus) for page in corpus}

    while True:
        new_ranks = {}

        for current_page in corpus:
            new_rank = (1 - damping_factor) / len(corpus)

            for page, links in corpus.items():
                if current_page in links:
                    new_rank += damping_factor * (ranks[page] / len(corpus[page]))
                elif not links:
                    # If `page` has no links,
                    # we can pretend it has links to all pages in `corpus`
                    new_rank += damping_factor * (ranks[page] / len(corpus))

            new_ranks[current_page] = new_rank

        # If all PR values converge
        if all(abs(new_ranks[page] - ranks[page]) <= 0.001 for page in corpus):
            return new_ranks
        ranks = new_ranks


if __name__ == "__main__":
    main()
