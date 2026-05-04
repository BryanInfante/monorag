import json
import pytest
from unittest.mock import MagicMock, patch
from rag_core import mcp_server

@pytest.fixture(autouse=True)
def clear_cache():
    mcp_server._instances.clear()

@pytest.mark.asyncio
async def test_tools_registered():
    tools = [tool.name for tool in await mcp_server.mcp.list_tools()]
    expected = ["search", "ask", "index_file", "index_directory", "list_collections", "create_collection", "delete_collection", "clear_history"]
    for tool in expected:
        assert tool in tools

@patch("rag_core.mcp_server.RAGModule")
def test_search_success(mock_rag):
    mock_instance = MagicMock()
    mock_rag.return_value = mock_instance
    mock_instance.search.return_value = [{"text": "hola", "metadata": {}}]
    
    result = mcp_server.search(query="test", collection="test_col")
    
    assert json.loads(result) == [{"text": "hola", "metadata": {}}]
    mock_instance.search.assert_called_with("test", 5)

def test_search_empty_params():
    result = mcp_server.search(query="", collection="test")
    assert "Error" in result

@patch("rag_core.mcp_server.RAGModule")
def test_ask_success(mock_rag):
    mock_instance = MagicMock()
    mock_rag.return_value = mock_instance
    mock_instance.ask.return_value = "Respuesta"
    
    result = mcp_server.ask(question="¿hola?", collection="test_col")
    
    assert result == "Respuesta"
    mock_instance.ask.assert_called_with("¿hola?", 5)

@patch("rag_core.mcp_server.RAGModule")
def test_index_file_success(mock_rag):
    mock_instance = MagicMock()
    mock_rag.return_value = mock_instance
    mock_instance.add_file.return_value = 10
    
    result = mcp_server.index_file(path="doc.pdf", collection="test_col")
    
    assert "10" in result
    mock_instance.add_file.assert_called_with("doc.pdf")

@patch("chromadb.PersistentClient")
def test_list_collections(mock_client_cls):
    mock_client = MagicMock()
    mock_col1 = MagicMock()
    mock_col1.name = "col1"
    mock_col2 = MagicMock()
    mock_col2.name = "col2"
    mock_client.list_collections.return_value = [mock_col1, mock_col2]
    mock_client_cls.return_value = mock_client
    
    result = mcp_server.list_collections()
    
    assert json.loads(result) == ["col1", "col2"]
    assert "col1" in result

@patch("rag_core.mcp_server.RAGModule")
def test_create_collection(mock_rag):
    result = mcp_server.create_collection(name="nueva_col")
    
    assert "correctamente" in result
    assert "nueva_col" in mcp_server._instances

@patch("rag_core.mcp_server.RAGModule")
def test_delete_collection_cached(mock_rag):
    mock_instance = MagicMock()
    mcp_server._instances["col"] = mock_instance
    
    result = mcp_server.delete_collection(name="col")
    
    assert "eliminada" in result
    assert "col" not in mcp_server._instances
    mock_instance.delete_collection.assert_called_once()

@patch("rag_core.mcp_server.RAGModule")
def test_clear_history_success(mock_rag):
    mock_instance = MagicMock()
    mcp_server._instances["col"] = mock_instance
    
    result = mcp_server.clear_history(collection="col")
    
    assert "borrado" in result
    mock_instance.clear_history.assert_called_once()

def test_clear_history_no_session():
    result = mcp_server.clear_history(collection="no_existe")
    assert "Error" in result
