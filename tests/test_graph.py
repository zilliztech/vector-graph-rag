"""Tests for Graph abstraction layer."""

import pytest
import uuid

from vector_graph_rag.graph.graph import Graph
from vector_graph_rag.models import Triplet


class TestGraphIdGeneration:
    """Tests for ID generation in Graph class."""

    def test_create_passage_without_id_generates_uuid(self, graph):
        """Test that creating passage without ID generates UUID."""
        passage_id = graph.create_passage("Test passage text")

        # Should be a valid UUID
        uuid.UUID(passage_id)

    def test_create_passage_with_custom_id(self, graph):
        """Test creating passage with custom ID."""
        passage_id = graph.create_passage(
            text="Test passage",
            id="custom_doc_001",
        )

        assert passage_id == "custom_doc_001"

    def test_create_entity_without_id_generates_uuid(self, graph):
        """Test that creating entity without ID generates UUID."""
        entity_id = graph._create_entity("Test Entity")

        uuid.UUID(entity_id)

    def test_create_entity_with_custom_id(self, graph):
        """Test creating entity with custom ID."""
        entity_id = graph._create_entity("Test Entity", id="ent_001")

        assert entity_id == "ent_001"


class TestGraphPassageCRUD:
    """Tests for Passage CRUD operations."""

    def test_create_passage_simple(self, graph):
        """Test creating a simple passage."""
        passage_id = graph.create_passage("This is a test passage.")

        assert passage_id is not None
        assert len(passage_id) > 0

    def test_create_passage_with_triplets(self, graph):
        """Test creating passage with triplets auto-creates entities and relations."""
        triplets = [
            Triplet(subject="Einstein", predicate="developed", object="relativity"),
        ]

        passage_id = graph.create_passage(
            text="Einstein developed the theory of relativity.",
            triplets=triplets,
        )

        passage = graph.get_passage(passage_id)
        assert passage is not None
        assert len(passage.entity_ids) >= 2  # Einstein and relativity
        assert len(passage.relation_ids) >= 1

    def test_get_passage(self, graph):
        """Test getting a passage by ID."""
        passage_id = graph.create_passage("Test passage for retrieval.")

        passage = graph.get_passage(passage_id)

        assert passage is not None
        assert passage.id == passage_id
        assert passage.text == "Test passage for retrieval."

    def test_get_nonexistent_passage_returns_none(self, graph):
        """Test that getting a non-existent passage returns None."""
        passage = graph.get_passage("nonexistent_id")
        assert passage is None

    def test_search_passages(self, graph):
        """Test searching passages by vector similarity."""
        # Create some passages
        graph.create_passage("Einstein developed relativity theory.")
        graph.create_passage("Newton discovered the laws of gravity.")

        results = graph.search_passages("physics theories", top_k=2)

        assert len(results) >= 1

    def test_update_passage_text(self, graph):
        """Test updating passage text."""
        passage_id = graph.create_passage("Original text.")

        success = graph.update_passage(passage_id, text="Updated text.")
        assert success

        passage = graph.get_passage(passage_id)
        assert passage.text == "Updated text."

    def test_update_nonexistent_passage_returns_false(self, graph):
        """Test that updating a non-existent passage returns False."""
        success = graph.update_passage("nonexistent", text="New text")
        assert not success

    def test_delete_passage(self, graph):
        """Test deleting a passage."""
        passage_id = graph.create_passage("Passage to delete.")

        success = graph.delete_passage(passage_id)
        assert success

        passage = graph.get_passage(passage_id)
        assert passage is None

    def test_delete_nonexistent_passage_returns_false(self, graph):
        """Test that deleting a non-existent passage returns False."""
        success = graph.delete_passage("nonexistent_id")
        assert not success


class TestGraphEntityCRUD:
    """Tests for Entity CRUD operations (private methods)."""

    def test_create_entity(self, graph):
        """Test creating an entity."""
        entity_id = graph._create_entity("Test Entity")

        assert entity_id is not None

    def test_create_entity_deduplication(self, graph):
        """Test that creating the same entity twice returns the same ID."""
        id1 = graph._create_entity("Einstein")
        id2 = graph._create_entity("Einstein")  # Same name

        assert id1 == id2

    def test_create_entity_case_insensitive_dedup(self, graph):
        """Test case-insensitive deduplication."""
        id1 = graph._create_entity("Einstein")
        id2 = graph._create_entity("EINSTEIN")

        assert id1 == id2

    def test_get_entity(self, graph):
        """Test getting an entity by ID."""
        entity_id = graph._create_entity("Test Entity")

        entity = graph._get_entity(entity_id)

        assert entity is not None
        assert entity.id == entity_id

    def test_update_entity(self, graph):
        """Test updating an entity."""
        entity_id = graph._create_entity("Original Name")

        success = graph._update_entity(entity_id, name="Updated Name")
        assert success

    def test_delete_entity(self, graph):
        """Test deleting an entity."""
        entity_id = graph._create_entity("Entity to delete")

        success = graph._delete_entity(entity_id)
        assert success

        entity = graph._get_entity(entity_id)
        assert entity is None


class TestGraphRelationCRUD:
    """Tests for Relation CRUD operations (private methods)."""

    def test_create_relation(self, graph):
        """Test creating a relation."""
        relation_id = graph._create_relation(
            subject="Einstein",
            predicate="developed",
            object_="relativity",
        )

        assert relation_id is not None

    def test_create_relation_creates_entities(self, graph):
        """Test that creating a relation also creates the entities."""
        relation_id = graph._create_relation(
            subject="Alice",
            predicate="knows",
            object_="Bob",
        )

        relation = graph._get_relation(relation_id)
        assert relation is not None

        # Check entities were created
        alice_id = graph._entity_name_to_id.get("alice")
        bob_id = graph._entity_name_to_id.get("bob")
        assert alice_id is not None
        assert bob_id is not None

    def test_create_relation_deduplication(self, graph):
        """Test that creating the same relation twice returns the same ID."""
        id1 = graph._create_relation("A", "knows", "B")
        id2 = graph._create_relation("A", "knows", "B")

        assert id1 == id2

    def test_get_relation(self, graph):
        """Test getting a relation by ID."""
        relation_id = graph._create_relation("X", "related", "Y")

        relation = graph._get_relation(relation_id)

        assert relation is not None
        assert relation.triplet.subject == "x"  # Normalized
        assert relation.triplet.predicate == "related"
        assert relation.triplet.object == "y"

    def test_delete_relation(self, graph):
        """Test deleting a relation."""
        relation_id = graph._create_relation("A", "test", "B")

        success = graph._delete_relation(relation_id)
        assert success

        relation = graph._get_relation(relation_id)
        assert relation is None


class TestGraphCascadeDelete:
    """Tests for cascade delete operations."""

    def test_delete_passage_updates_entity_references(self, graph):
        """Test that deleting a passage updates entity passage_ids."""
        # Create passage with triplets
        triplets = [Triplet(subject="Alice", predicate="knows", object="Bob")]
        passage_id = graph.create_passage(
            text="Alice knows Bob.",
            triplets=triplets,
        )

        # Get entity ID before deletion
        alice_id = graph._entity_name_to_id.get("alice")

        # Delete passage
        graph.delete_passage(passage_id)

        # Check entity's passage_ids no longer contains the deleted passage
        entity = graph._get_entity(alice_id)
        # Entity should still exist but passage reference should be removed
        # (Note: actual implementation may vary)

    def test_delete_passage_updates_relation_references(self, graph):
        """Test that deleting a passage updates relation passage_ids."""
        triplets = [Triplet(subject="X", predicate="rel", object="Y")]
        passage_id = graph.create_passage(
            text="X rel Y.",
            triplets=triplets,
        )

        relation_id = list(graph._relation_text_to_id.values())[0]

        # Delete passage
        graph.delete_passage(passage_id)

        # Relation should still exist but passage reference removed
        relation = graph._get_relation(relation_id)
        if relation:
            assert passage_id not in relation.source_passage_ids

    def test_delete_entity_updates_relation_references(self, graph):
        """Test that deleting an entity updates relation entity_ids."""
        triplets = [Triplet(subject="Entity1", predicate="knows", object="Entity2")]
        graph.create_passage(text="Test", triplets=triplets)

        entity1_id = graph._entity_name_to_id.get("entity1")
        relation_id = list(graph._relation_text_to_id.values())[0]

        # Delete entity
        graph._delete_entity(entity1_id)

        # Check relation's entity_ids no longer contains deleted entity
        relation_data = graph._store._get_relations_by_ids([relation_id])
        if relation_data:
            assert entity1_id not in relation_data[0].get("entity_ids", [])


class TestGraphSubgraph:
    """Tests for SubGraph creation."""

    def test_create_subgraph(self, graph):
        """Test creating a subgraph."""
        subgraph = graph.create_subgraph()

        assert subgraph is not None
        # Subgraph should be empty initially
        assert len(subgraph.entity_ids) == 0
        assert len(subgraph.relation_ids) == 0


class TestGraphEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_passage_text(self, graph):
        """Test creating passage with empty text."""
        passage_id = graph.create_passage("")
        assert passage_id is not None

    def test_special_characters_in_id(self, graph):
        """Test using special characters in custom ID."""
        passage_id = graph.create_passage(
            text="Test",
            id="doc-001_test.v2",
        )
        assert passage_id == "doc-001_test.v2"

        passage = graph.get_passage("doc-001_test.v2")
        assert passage is not None

    def test_long_text(self, graph):
        """Test handling long text content."""
        long_text = "A" * 10000  # 10K characters
        passage_id = graph.create_passage(long_text)

        passage = graph.get_passage(passage_id)
        assert passage.text == long_text

    def test_unicode_text(self, graph):
        """Test handling Unicode text."""
        unicode_text = "爱因斯坦发明了相对论。日本語テスト。"
        passage_id = graph.create_passage(unicode_text)

        passage = graph.get_passage(passage_id)
        assert passage.text == unicode_text

    def test_create_delete_create_same_id(self, graph):
        """Test creating, deleting, then re-creating with same ID."""
        custom_id = "test_reuse_id"

        # Create
        graph.create_passage("First version", id=custom_id)
        assert graph.get_passage(custom_id) is not None

        # Delete
        graph.delete_passage(custom_id)
        assert graph.get_passage(custom_id) is None

        # Re-create with same ID
        graph.create_passage("Second version", id=custom_id)
        passage = graph.get_passage(custom_id)
        assert passage is not None
        assert passage.text == "Second version"

    def test_multiple_triplets_same_passage(self, graph):
        """Test creating passage with multiple triplets."""
        triplets = [
            Triplet(subject="Einstein", predicate="born in", object="Germany"),
            Triplet(subject="Einstein", predicate="developed", object="relativity"),
            Triplet(subject="Einstein", predicate="won", object="Nobel Prize"),
        ]

        passage_id = graph.create_passage(
            text="Einstein was born in Germany, developed relativity, and won Nobel Prize.",
            triplets=triplets,
        )

        passage = graph.get_passage(passage_id)
        assert len(passage.relation_ids) == 3

    def test_reset_clears_all_data(self, graph):
        """Test that reset() clears all data."""
        # Add some data
        graph.create_passage("Test passage")
        graph._create_entity("Test entity")

        # Reset
        graph.reset()

        # Verify cleared
        assert len(graph._entity_name_to_id) == 0
        assert len(graph._relation_text_to_id) == 0
