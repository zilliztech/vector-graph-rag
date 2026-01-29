"""Tests for MilvusStore CRUD operations."""

import pytest
import uuid

from vector_graph_rag.storage.milvus import MilvusStore, generate_id


class TestGenerateId:
    """Tests for ID generation."""

    def test_generate_id_returns_string(self):
        """Test that generate_id returns a string."""
        id_ = generate_id()
        assert isinstance(id_, str)

    def test_generate_id_is_valid_uuid(self):
        """Test that generate_id returns a valid UUID."""
        id_ = generate_id()
        # Should not raise
        uuid.UUID(id_)

    def test_generate_id_unique(self):
        """Test that generate_id returns unique IDs."""
        ids = [generate_id() for _ in range(100)]
        assert len(set(ids)) == 100


class TestMilvusStoreIdTypes:
    """Tests for ID type handling (str IDs)."""

    def test_insert_entities_with_str_ids(self, milvus_store):
        """Test inserting entities with string IDs."""
        ids = ["entity_001", "entity_002", "entity_003"]
        texts = ["Einstein", "Relativity", "Physics"]
        embeddings = [[0.1] * 1536, [0.2] * 1536, [0.3] * 1536]

        result_ids = milvus_store._insert_entities(
            texts, ids=ids, embeddings=embeddings
        )

        assert result_ids == ids

    def test_insert_entities_without_ids_generates_uuids(self, milvus_store):
        """Test that entities get UUID IDs when not provided."""
        texts = ["Einstein", "Relativity"]
        embeddings = [[0.1] * 1536, [0.2] * 1536]

        result_ids = milvus_store._insert_entities(
            texts, embeddings=embeddings
        )

        assert len(result_ids) == 2
        for id_ in result_ids:
            # Should be valid UUIDs
            uuid.UUID(id_)

    def test_insert_and_get_entities_by_ids(self, milvus_store):
        """Test inserting and retrieving entities by IDs."""
        ids = ["ent_1", "ent_2"]
        texts = ["Alice", "Bob"]
        embeddings = [[0.1] * 1536, [0.2] * 1536]
        metadatas = [
            {"relation_ids": ["rel_1"], "passage_ids": ["p_1"]},
            {"relation_ids": ["rel_2"], "passage_ids": ["p_2"]},
        ]

        milvus_store._insert_entities(
            texts, ids=ids, embeddings=embeddings, metadatas=metadatas
        )

        results = milvus_store._get_entities_by_ids(ids)

        assert len(results) == 2
        id_to_data = {r["id"]: r for r in results}
        assert "ent_1" in id_to_data
        assert id_to_data["ent_1"]["text"] == "Alice"
        assert id_to_data["ent_1"]["relation_ids"] == ["rel_1"]

    def test_insert_relations_with_triplet_metadata(self, milvus_store):
        """Test inserting relations with structured triplet metadata."""
        ids = ["rel_001"]
        texts = ["einstein developed relativity"]
        embeddings = [[0.1] * 1536]
        metadatas = [{
            "entity_ids": ["ent_1", "ent_2"],
            "passage_ids": ["p_1"],
            "subject": "einstein",
            "predicate": "developed",
            "object": "relativity",
        }]

        milvus_store._insert_relations(
            texts, ids=ids, embeddings=embeddings, metadatas=metadatas
        )

        results = milvus_store._get_relations_by_ids(ids)

        assert len(results) == 1
        assert results[0]["subject"] == "einstein"
        assert results[0]["predicate"] == "developed"
        assert results[0]["object"] == "relativity"

    def test_insert_passages_with_str_ids(self, milvus_store):
        """Test inserting passages with string IDs."""
        ids = ["doc_001", "doc_002"]
        texts = ["First passage text.", "Second passage text."]
        embeddings = [[0.1] * 1536, [0.2] * 1536]
        metadatas = [
            {"entity_ids": ["e1"], "relation_ids": ["r1"]},
            {"entity_ids": ["e2"], "relation_ids": ["r2"]},
        ]

        result_ids = milvus_store.insert_passages(
            texts, ids=ids, embeddings=embeddings, metadatas=metadatas
        )

        assert result_ids == ids

        results = milvus_store.get_passages_by_ids(ids)
        assert len(results) == 2


class TestMilvusStoreUpdate:
    """Tests for update operations."""

    def test_update_entity_text(self, milvus_store):
        """Test updating entity text."""
        # Insert
        ids = milvus_store._insert_entities(
            ["Original Name"],
            embeddings=[[0.1] * 1536],
        )
        entity_id = ids[0]

        # Update
        success = milvus_store._update_entity(entity_id, text="Updated Name")
        assert success

        # Verify
        results = milvus_store._get_entities_by_ids([entity_id])
        assert results[0]["text"] == "Updated Name"

    def test_update_entity_metadata(self, milvus_store):
        """Test updating entity metadata fields."""
        ids = milvus_store._insert_entities(
            ["Test Entity"],
            embeddings=[[0.1] * 1536],
            metadatas=[{"relation_ids": ["r1"], "passage_ids": ["p1"]}],
        )
        entity_id = ids[0]

        # Update relation_ids
        success = milvus_store._update_entity(
            entity_id,
            relation_ids=["r1", "r2", "r3"],
        )
        assert success

        results = milvus_store._get_entities_by_ids([entity_id])
        assert "r2" in results[0]["relation_ids"]
        assert "r3" in results[0]["relation_ids"]

    def test_update_nonexistent_entity_returns_false(self, milvus_store):
        """Test that updating a non-existent entity returns False."""
        success = milvus_store._update_entity(
            "nonexistent_id",
            text="New Text",
        )
        assert not success

    def test_update_relation(self, milvus_store):
        """Test updating a relation."""
        ids = milvus_store._insert_relations(
            ["alice knows bob"],
            embeddings=[[0.1] * 1536],
            metadatas=[{
                "entity_ids": ["e1", "e2"],
                "subject": "alice",
                "predicate": "knows",
                "object": "bob",
            }],
        )
        relation_id = ids[0]

        # Update
        success = milvus_store._update_relation(
            relation_id,
            predicate="works with",
            text="alice works with bob",
        )
        assert success

        results = milvus_store._get_relations_by_ids([relation_id])
        assert results[0]["predicate"] == "works with"
        assert results[0]["text"] == "alice works with bob"

    def test_update_passage(self, milvus_store):
        """Test updating a passage."""
        ids = milvus_store.insert_passages(
            ["Original passage text."],
            embeddings=[[0.1] * 1536],
        )
        passage_id = ids[0]

        # Update
        success = milvus_store.update_passage(
            passage_id,
            text="Updated passage text.",
        )
        assert success

        results = milvus_store.get_passages_by_ids([passage_id])
        assert results[0]["text"] == "Updated passage text."


class TestMilvusStoreDelete:
    """Tests for delete operations."""

    def test_delete_entity(self, milvus_store):
        """Test deleting an entity."""
        ids = milvus_store._insert_entities(
            ["Entity to delete"],
            embeddings=[[0.1] * 1536],
        )
        entity_id = ids[0]

        # Verify exists
        assert len(milvus_store._get_entities_by_ids([entity_id])) == 1

        # Delete
        success = milvus_store._delete_entity(entity_id)
        assert success

        # Verify deleted
        assert len(milvus_store._get_entities_by_ids([entity_id])) == 0

    def test_delete_nonexistent_entity_returns_false(self, milvus_store):
        """Test that deleting a non-existent entity returns False."""
        success = milvus_store._delete_entity("nonexistent_id")
        assert not success

    def test_delete_relation(self, milvus_store):
        """Test deleting a relation."""
        ids = milvus_store._insert_relations(
            ["relation to delete"],
            embeddings=[[0.1] * 1536],
        )
        relation_id = ids[0]

        success = milvus_store._delete_relation(relation_id)
        assert success

        assert len(milvus_store._get_relations_by_ids([relation_id])) == 0

    def test_delete_passage(self, milvus_store):
        """Test deleting a passage."""
        ids = milvus_store.insert_passages(
            ["Passage to delete"],
            embeddings=[[0.1] * 1536],
        )
        passage_id = ids[0]

        success = milvus_store.delete_passage(passage_id)
        assert success

        assert len(milvus_store.get_passages_by_ids([passage_id])) == 0

    def test_delete_multiple_entities(self, milvus_store):
        """Test batch deleting entities."""
        ids = milvus_store._insert_entities(
            ["Entity 1", "Entity 2", "Entity 3"],
            embeddings=[[0.1] * 1536, [0.2] * 1536, [0.3] * 1536],
        )

        count = milvus_store._delete_entities(ids)
        assert count == 3

        assert len(milvus_store._get_entities_by_ids(ids)) == 0

    def test_delete_multiple_relations(self, milvus_store):
        """Test batch deleting relations."""
        ids = milvus_store._insert_relations(
            ["rel 1", "rel 2"],
            embeddings=[[0.1] * 1536, [0.2] * 1536],
        )

        count = milvus_store._delete_relations(ids)
        assert count == 2

    def test_delete_multiple_passages(self, milvus_store):
        """Test batch deleting passages."""
        ids = milvus_store.insert_passages(
            ["passage 1", "passage 2"],
            embeddings=[[0.1] * 1536, [0.2] * 1536],
        )

        count = milvus_store.delete_passages(ids)
        assert count == 2

    def test_delete_empty_list(self, milvus_store):
        """Test deleting empty list returns 0."""
        assert milvus_store._delete_entities([]) == 0
        assert milvus_store._delete_relations([]) == 0
        assert milvus_store.delete_passages([]) == 0


class TestMilvusStoreSearch:
    """Tests for search operations."""

    def test_search_entities(self, milvus_store):
        """Test searching entities by vector similarity."""
        milvus_store._insert_entities(
            ["Einstein", "Newton", "Darwin"],
            embeddings=[[0.1] * 1536, [0.2] * 1536, [0.3] * 1536],
        )

        results = milvus_store._search_entities(
            [[0.1] * 1536],
            top_k=2,
        )

        assert len(results) == 1  # One query
        assert len(results[0]) >= 1  # At least one result

    def test_search_relations(self, milvus_store):
        """Test searching relations by vector similarity."""
        milvus_store._insert_relations(
            ["einstein developed relativity", "newton discovered gravity"],
            embeddings=[[0.1] * 1536, [0.9] * 1536],
        )

        results = milvus_store._search_relations(
            [0.1] * 1536,
            top_k=2,
        )

        assert len(results) >= 1

    def test_search_passages(self, milvus_store):
        """Test searching passages by vector similarity."""
        milvus_store.insert_passages(
            ["First document about physics.", "Second document about biology."],
            embeddings=[[0.1] * 1536, [0.9] * 1536],
        )

        results = milvus_store.search_passages(
            [0.1] * 1536,
            top_k=2,
        )

        assert len(results) >= 1
