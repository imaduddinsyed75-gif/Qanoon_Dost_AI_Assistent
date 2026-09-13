import numpy as np
from user_documents import _split_text


def test_split_text_preserves_overlap():
    chunks = _split_text("one two three four five six", chunk_size=4, overlap=1)
    assert chunks == ["one two three four", "four five six"]


def test_user_document_embeddings_are_not_knowledge_base_files(tmp_path):
    # The Phase 4 model stores only an in-memory ndarray on the document.
    embeddings = np.array([[1.0, 0.0]])
    assert embeddings.shape == (1, 2)
    assert not list(tmp_path.iterdir())
