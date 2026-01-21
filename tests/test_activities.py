"""Tests for activity tools."""

import base64
import json
from unittest.mock import MagicMock

import pytest

from garmin_connect_mcp.tools.activities import download_activity


@pytest.fixture
def mock_context():
    """Create a mock FastMCP context with a mocked Garmin client."""
    mock_client = MagicMock()
    mock_ctx = MagicMock()
    mock_ctx.get_state.return_value = mock_client
    return mock_ctx, mock_client


@pytest.mark.asyncio
async def test_download_activity_gpx(mock_context):
    """Test downloading activity as GPX format."""
    mock_ctx, mock_client = mock_context

    # Mock GPX content
    gpx_content = b'<?xml version="1.0"?><gpx version="1.1">test</gpx>'
    mock_client.safe_call.return_value = gpx_content

    result = await download_activity(
        activity_id=12345,
        file_format="gpx",
        ctx=mock_ctx,
    )

    # Parse response
    response = json.loads(result)

    assert "data" in response
    assert response["data"]["activity_id"] == 12345
    assert response["data"]["format"] == "gpx"
    assert response["data"]["filename"] == "activity_12345.gpx"
    assert response["data"]["size_bytes"] == len(gpx_content)

    # Verify base64 encoding
    decoded = base64.b64decode(response["data"]["content_base64"])
    assert decoded == gpx_content


@pytest.mark.asyncio
async def test_download_activity_fit(mock_context):
    """Test downloading activity as FIT (original/zip) format."""
    mock_ctx, mock_client = mock_context

    # Mock ZIP content (FIT files are returned as ZIP)
    zip_content = b"PK\x03\x04test_zip_content"
    mock_client.safe_call.return_value = zip_content

    result = await download_activity(
        activity_id=12345,
        file_format="fit",
        ctx=mock_ctx,
    )

    response = json.loads(result)

    assert response["data"]["format"] == "fit"
    assert response["data"]["filename"] == "activity_12345.zip"
    assert response["data"]["size_bytes"] == len(zip_content)

    decoded = base64.b64decode(response["data"]["content_base64"])
    assert decoded == zip_content


@pytest.mark.asyncio
async def test_download_activity_tcx(mock_context):
    """Test downloading activity as TCX format."""
    mock_ctx, mock_client = mock_context

    tcx_content = b'<?xml version="1.0"?><TrainingCenterDatabase>test</TrainingCenterDatabase>'
    mock_client.safe_call.return_value = tcx_content

    result = await download_activity(
        activity_id=12345,
        file_format="tcx",
        ctx=mock_ctx,
    )

    response = json.loads(result)

    assert response["data"]["format"] == "tcx"
    assert response["data"]["filename"] == "activity_12345.tcx"


@pytest.mark.asyncio
async def test_download_activity_csv(mock_context):
    """Test downloading activity as CSV format."""
    mock_ctx, mock_client = mock_context

    csv_content = b"lap,distance,time\n1,1000,300\n2,1000,305"
    mock_client.safe_call.return_value = csv_content

    result = await download_activity(
        activity_id=12345,
        file_format="csv",
        ctx=mock_ctx,
    )

    response = json.loads(result)

    assert response["data"]["format"] == "csv"
    assert response["data"]["filename"] == "activity_12345.csv"


@pytest.mark.asyncio
async def test_download_activity_kml(mock_context):
    """Test downloading activity as KML format."""
    mock_ctx, mock_client = mock_context

    kml_content = b'<?xml version="1.0"?><kml>test</kml>'
    mock_client.safe_call.return_value = kml_content

    result = await download_activity(
        activity_id=12345,
        file_format="kml",
        ctx=mock_ctx,
    )

    response = json.loads(result)

    assert response["data"]["format"] == "kml"
    assert response["data"]["filename"] == "activity_12345.kml"


@pytest.mark.asyncio
async def test_download_activity_invalid_format(mock_context):
    """Test that invalid format returns validation error."""
    mock_ctx, mock_client = mock_context

    result = await download_activity(
        activity_id=12345,
        file_format="invalid",  # type: ignore
        ctx=mock_ctx,
    )

    response = json.loads(result)

    assert "error" in response
    assert response["error"]["type"] == "validation_error"
    assert "Invalid format" in response["error"]["message"]


@pytest.mark.asyncio
async def test_download_activity_empty_response(mock_context):
    """Test handling of empty response from Garmin API."""
    mock_ctx, mock_client = mock_context
    mock_client.safe_call.return_value = None

    result = await download_activity(
        activity_id=12345,
        file_format="gpx",
        ctx=mock_ctx,
    )

    response = json.loads(result)

    assert "error" in response
    assert response["error"]["type"] == "not_found"


@pytest.mark.asyncio
async def test_download_activity_api_error(mock_context):
    """Test handling of Garmin API errors."""
    from garmin_connect_mcp.client import GarminAPIError

    mock_ctx, mock_client = mock_context
    mock_client.safe_call.side_effect = GarminAPIError("Activity not found")

    result = await download_activity(
        activity_id=99999,
        file_format="gpx",
        ctx=mock_ctx,
    )

    response = json.loads(result)

    assert "error" in response
    assert response["error"]["type"] == "api_error"
    assert "Activity not found" in response["error"]["message"]


@pytest.mark.asyncio
async def test_download_activity_metadata(mock_context):
    """Test that response includes proper metadata."""
    mock_ctx, mock_client = mock_context
    mock_client.safe_call.return_value = b"test content"

    result = await download_activity(
        activity_id=12345,
        file_format="gpx",
        ctx=mock_ctx,
    )

    response = json.loads(result)

    assert "metadata" in response
    assert response["metadata"]["query_type"] == "activity_download"
    assert response["metadata"]["activity_id"] == 12345
    assert response["metadata"]["format"] == "gpx"
    assert "fetched_at" in response["metadata"]


@pytest.mark.asyncio
async def test_download_activity_analysis_insights(mock_context):
    """Test that response includes helpful analysis insights."""
    mock_ctx, mock_client = mock_context
    mock_client.safe_call.return_value = b"test content"

    result = await download_activity(
        activity_id=12345,
        file_format="gpx",
        ctx=mock_ctx,
    )

    response = json.loads(result)

    assert "analysis" in response
    assert "insights" in response["analysis"]
    # Should include download info and decode instructions
    insights = response["analysis"]["insights"]
    assert any("Downloaded" in i for i in insights)
    assert any("base64" in i.lower() for i in insights)
