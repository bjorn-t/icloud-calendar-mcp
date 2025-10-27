#!/usr/bin/env python3
"""
iCloud Calendar MCP Server

This MCP server provides integration with Apple iCloud Calendars via CalDAV protocol.
Features per-calendar permission management for granular read/write control.

DISCLAIMER:
This is an unofficial, third-party open-source tool. It is not affiliated with,
endorsed by, or sponsored by Apple Inc. This tool uses the standard CalDAV protocol
(RFC 4791) to access calendar data via Apple's publicly documented CalDAV endpoint.
iCloud and Apple Calendar are trademarks of Apple Inc., registered in the U.S. and
other countries.

Authentication Requirements:
- Apple ID (iCloud email)
- App-specific password (generate at https://appleid.apple.com)

Setup:
1. Generate an app-specific password for your Apple ID
2. Set environment variables:
   - ICLOUD_USERNAME: Your Apple ID (e.g., user@icloud.com)
   - ICLOUD_PASSWORD: Your app-specific password
3. Run: python icloud_calendar_mcp.py

Per-Calendar Permissions:
- Permissions are stored in ~/.icloud_calendar_permissions.json
- Use icloud_set_calendar_permissions to configure access
- Read/write operations are validated against these permissions
"""

import os
import json
from datetime import datetime, timedelta, timezone
from typing import Optional, List, Dict, Any
from pathlib import Path
from enum import Enum
from functools import lru_cache

import caldav
from caldav.lib.error import AuthorizationError, NotFoundError
from icalendar import Calendar as iCalendar
from pydantic import BaseModel, Field, ConfigDict
from mcp.server.fastmcp import FastMCP
from dotenv import load_dotenv

# Load environment variables from .env file (if it exists)
load_dotenv()

# Initialize MCP server
mcp = FastMCP("icloud_calendar_mcp")

# Constants
CHARACTER_LIMIT = 25000
CALDAV_URL = "https://caldav.icloud.com/"
PERMISSIONS_FILE = Path.home() / ".icloud_calendar_permissions.json"
MAX_CALENDAR_SAMPLE_EVENTS = 3
TRUNCATION_MESSAGE_RESERVE = 500
DEFAULT_DATE_RANGE_DAYS = 30

# ============================================================================
# Configuration and Permissions Management
# ============================================================================


class CalendarPermissions:
    """Manages per-calendar read/write permissions."""

    def __init__(self, permissions_file: Path = PERMISSIONS_FILE):
        self.permissions_file = permissions_file
        self.permissions: Dict[str, Dict[str, bool]] = self._load_permissions()

    def _load_permissions(self) -> Dict[str, Dict[str, bool]]:
        """Load permissions from file."""
        if self.permissions_file.exists():
            try:
                with open(self.permissions_file, "r") as f:
                    return json.load(f)
            except Exception:
                return {}
        return {}

    def _save_permissions(self):
        """Save permissions to file."""
        self.permissions_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.permissions_file, "w") as f:
            json.dump(self.permissions, f, indent=2)

    def get_calendar_permissions(self, calendar_name: str) -> Dict[str, bool]:
        """Get permissions for a specific calendar."""
        return self.permissions.get(
            calendar_name, {"read": True, "write": False}  # Default: read-only
        )

    def set_calendar_permissions(
        self, calendar_name: str, read: bool = True, write: bool = False
    ):
        """Set permissions for a specific calendar."""
        self.permissions[calendar_name] = {"read": read, "write": write}
        self._save_permissions()

    def can_read(self, calendar_name: str) -> bool:
        """Check if calendar has read permission."""
        return self.get_calendar_permissions(calendar_name).get("read", True)

    def can_write(self, calendar_name: str) -> bool:
        """Check if calendar has write permission."""
        return self.get_calendar_permissions(calendar_name).get("write", False)

    def get_all_permissions(self) -> Dict[str, Dict[str, bool]]:
        """Get all calendar permissions."""
        return self.permissions


# Global permissions manager
permissions_manager = CalendarPermissions()


# ============================================================================
# Pydantic Models for Input Validation
# ============================================================================


class ResponseFormat(str, Enum):
    """Output format for tool responses."""

    MARKDOWN = "markdown"
    JSON = "json"


class ListCalendarsInput(BaseModel):
    """Input for listing calendars."""

    model_config = ConfigDict(str_strip_whitespace=True, validate_assignment=True)

    response_format: ResponseFormat = Field(
        default=ResponseFormat.MARKDOWN,
        description="Output format: 'markdown' for human-readable or 'json' for machine-readable",
    )


class GetEventsInput(BaseModel):
    """Input for getting calendar events."""

    model_config = ConfigDict(str_strip_whitespace=True, validate_assignment=True)

    calendar_name: str = Field(
        ...,
        description="Name of the calendar to fetch events from (e.g., 'Work', 'Personal', 'Home')",
        min_length=1,
    )
    start_date: Optional[str] = Field(
        default=None,
        description="Start date in ISO format YYYY-MM-DD or ISO datetime (e.g., '2025-01-01' or '2025-01-01T00:00:00'). Defaults to today.",
    )
    end_date: Optional[str] = Field(
        default=None,
        description="End date in ISO format YYYY-MM-DD or ISO datetime (e.g., '2025-12-31' or '2025-12-31T23:59:59'). Defaults to 30 days from start.",
    )
    limit: Optional[int] = Field(
        default=50,
        description="Maximum number of events to return",
        ge=1,
        le=250,
    )
    response_format: ResponseFormat = Field(
        default=ResponseFormat.MARKDOWN,
        description="Output format: 'markdown' for human-readable or 'json' for machine-readable",
    )


class SearchEventsInput(BaseModel):
    """Input for searching events across calendars."""

    model_config = ConfigDict(str_strip_whitespace=True, validate_assignment=True)

    query: str = Field(
        ...,
        description="Search query to match against event titles and descriptions (e.g., 'meeting', 'dentist', 'birthday')",
        min_length=1,
    )
    start_date: Optional[str] = Field(
        default=None,
        description="Start date in ISO format YYYY-MM-DD. Defaults to today.",
    )
    end_date: Optional[str] = Field(
        default=None,
        description="End date in ISO format YYYY-MM-DD. Defaults to 30 days from start.",
    )
    calendar_names: Optional[List[str]] = Field(
        default=None,
        description="List of calendar names to search in. If not provided, searches all readable calendars.",
    )
    limit: Optional[int] = Field(
        default=50, description="Maximum number of events to return", ge=1, le=250
    )
    response_format: ResponseFormat = Field(
        default=ResponseFormat.MARKDOWN,
        description="Output format: 'markdown' for human-readable or 'json' for machine-readable",
    )


class CreateEventInput(BaseModel):
    """Input for creating a calendar event."""

    model_config = ConfigDict(str_strip_whitespace=True, validate_assignment=True)

    calendar_name: str = Field(
        ...,
        description="Name of the calendar to create the event in (e.g., 'Work', 'Personal')",
        min_length=1,
    )
    title: str = Field(
        ...,
        description="Event title/summary (e.g., 'Team Meeting', 'Doctor Appointment')",
        min_length=1,
        max_length=500,
    )
    start: str = Field(
        ...,
        description="Start datetime in ISO format (e.g., '2025-01-15T10:00:00' or '2025-01-15T10:00:00-05:00')",
    )
    end: str = Field(
        ...,
        description="End datetime in ISO format (e.g., '2025-01-15T11:00:00' or '2025-01-15T11:00:00-05:00')",
    )
    description: Optional[str] = Field(
        default=None,
        description="Event description/notes",
        max_length=10000,
    )
    location: Optional[str] = Field(
        default=None,
        description="Event location (e.g., 'Conference Room A', '123 Main St')",
        max_length=500,
    )
    all_day: bool = Field(
        default=False,
        description="Whether this is an all-day event",
    )


class UpdateEventInput(BaseModel):
    """Input for updating a calendar event."""

    model_config = ConfigDict(str_strip_whitespace=True, validate_assignment=True)

    calendar_name: str = Field(
        ...,
        description="Name of the calendar containing the event",
        min_length=1,
    )
    event_uid: str = Field(
        ...,
        description="Unique identifier (UID) of the event to update",
        min_length=1,
    )
    title: Optional[str] = Field(
        default=None,
        description="New event title/summary",
        min_length=1,
        max_length=500,
    )
    start: Optional[str] = Field(
        default=None,
        description="New start datetime in ISO format",
    )
    end: Optional[str] = Field(
        default=None,
        description="New end datetime in ISO format",
    )
    description: Optional[str] = Field(
        default=None, description="New event description/notes", max_length=10000
    )
    location: Optional[str] = Field(
        default=None, description="New event location", max_length=500
    )


class DeleteEventInput(BaseModel):
    """Input for deleting a calendar event."""

    model_config = ConfigDict(str_strip_whitespace=True, validate_assignment=True)

    calendar_name: str = Field(
        ...,
        description="Name of the calendar containing the event",
        min_length=1,
    )
    event_uid: str = Field(
        ...,
        description="Unique identifier (UID) of the event to delete",
        min_length=1,
    )


class SetPermissionsInput(BaseModel):
    """Input for setting calendar permissions."""

    model_config = ConfigDict(str_strip_whitespace=True, validate_assignment=True)

    calendar_name: str = Field(
        ...,
        description="Name of the calendar to configure permissions for",
        min_length=1,
    )
    read: bool = Field(
        default=True,
        description="Allow read access to this calendar",
    )
    write: bool = Field(
        default=False,
        description="Allow write access (create/update/delete events) to this calendar",
    )


# ============================================================================
# Helper Functions
# ============================================================================


def get_caldav_client() -> caldav.DAVClient:
    """Get authenticated CalDAV client for iCloud."""
    username = os.environ.get("ICLOUD_USERNAME")
    password = os.environ.get("ICLOUD_PASSWORD")

    if not username or not password:
        raise ValueError(
            "Missing credentials. Please set ICLOUD_USERNAME and ICLOUD_PASSWORD environment variables.\n"
            "Generate an app-specific password at: https://appleid.apple.com"
        )

    try:
        client = caldav.DAVClient(url=CALDAV_URL, username=username, password=password)
        return client
    except Exception as e:
        raise RuntimeError(
            f"Failed to connect to iCloud CalDAV server: {str(e)}\n"
            "Ensure your credentials are correct and you're using an app-specific password."
        )


def parse_datetime(dt_string: str) -> datetime:
    """Parse ISO datetime string with flexible timezone handling."""
    try:
        # Try parsing with timezone
        return datetime.fromisoformat(dt_string)
    except ValueError:
        # Try parsing without timezone and assume local
        try:
            dt = datetime.fromisoformat(dt_string)
            # If no timezone info, assume local timezone
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt
        except ValueError:
            # Try parsing as date only
            from dateutil import parser

            return parser.parse(dt_string)


def format_datetime(dt: Any) -> str:
    """Format datetime object to human-readable string."""
    if isinstance(dt, datetime):
        return dt.strftime("%Y-%m-%d %H:%M:%S %Z")
    elif hasattr(dt, "dt"):
        return format_datetime(dt.dt)
    return str(dt)


def event_to_dict(event: Any) -> Dict[str, Any]:
    """Convert CalDAV event to dictionary.

    Only processes VEVENT components (calendar events).
    Explicitly skips VTODO components (reminders/tasks) and other component types.
    """
    try:
        cal = iCalendar.from_ical(event.data)
        for component in cal.walk():
            # Skip VTODO (reminders/tasks) and other non-event components
            if component.name == "VTODO":
                continue
            elif component.name == "VEVENT":
                event_dict = {
                    "uid": str(component.get("uid", "")),
                    "title": str(component.get("summary", "")),
                    "description": str(component.get("description", "")),
                    "location": str(component.get("location", "")),
                    "start": format_datetime(component.get("dtstart")),
                    "end": format_datetime(component.get("dtend")),
                    "created": format_datetime(component.get("created"))
                    if component.get("created")
                    else None,
                    "last_modified": format_datetime(component.get("last-modified"))
                    if component.get("last-modified")
                    else None,
                }
                return event_dict  # Early return after finding first VEVENT
    except Exception as e:
        return {
            "uid": "unknown",
            "error": f"Failed to parse event: {str(e)}",
            "raw_data": event.data if hasattr(event, "data") else str(event),
        }
    return {}


# Cache for calendar type detection (URL -> is_event_calendar result)
_calendar_type_cache: Dict[str, bool] = {}


def is_event_calendar(calendar: Any) -> bool:
    """Check if a calendar is an event calendar (not a Reminders/Tasks calendar).

    This function filters out Reminders calendars which contain VTODO components
    instead of VEVENT components. Results are cached by calendar URL for performance.

    Args:
        calendar: CalDAV calendar object

    Returns:
        bool: True if the calendar supports VEVENT (events), False if it's a VTODO calendar (reminders/tasks)
    """
    # Check cache first
    calendar_url = str(calendar.url) if hasattr(calendar, 'url') else ""
    if calendar_url in _calendar_type_cache:
        return _calendar_type_cache[calendar_url]

    # Perform the check
    try:
        # Try to get the supported calendar component set property
        # This is a standard CalDAV property that indicates what component types
        # a calendar collection supports (VEVENT, VTODO, VJOURNAL, etc.)
        try:
            # Attempt to get the supported-calendar-component-set property
            comp_types = calendar.get_supported_components()
            if comp_types:
                # If we get component types, check if VEVENT is supported and VTODO is not the only type
                has_vevent = 'VEVENT' in comp_types
                has_only_vtodo = comp_types == {'VTODO'} or comp_types == ['VTODO']
                result = has_vevent and not has_only_vtodo
                _calendar_type_cache[calendar_url] = result
                return result
        except (AttributeError, Exception):
            # Property not available or method doesn't exist, try alternative approach
            pass

        # Fallback: Try to fetch a sample event and check component types
        # This is a more expensive operation but more reliable
        try:
            # Get calendar's raw data to inspect component types
            events = calendar.events()
            if events:
                # Check the first few events to see what component types they contain
                for event in events[:MAX_CALENDAR_SAMPLE_EVENTS]:
                    try:
                        cal = iCalendar.from_ical(event.data)
                        for component in cal.walk():
                            if component.name == "VEVENT":
                                # Found a VEVENT component, this is an event calendar
                                result = True
                                _calendar_type_cache[calendar_url] = result
                                return result
                            elif component.name == "VTODO":
                                # Found only VTODO, this is a reminders/tasks calendar
                                result = False
                                _calendar_type_cache[calendar_url] = result
                                return result
                    except Exception:
                        continue
        except Exception:
            # If we can't fetch events, assume it's an event calendar
            # (better to include than exclude in case of errors)
            pass

        # Default: If we can't determine, assume it's an event calendar
        # This ensures we don't accidentally filter out valid calendars
        result = True
        _calendar_type_cache[calendar_url] = result
        return result

    except Exception:
        # On any error, default to including the calendar
        result = True
        _calendar_type_cache[calendar_url] = result
        return result


def format_events_markdown(events: List[Dict[str, Any]]) -> str:
    """Format events as markdown."""
    if not events:
        return "No events found."

    output = []
    for event in events:
        output.append(f"### {event.get('title', 'Untitled Event')}")
        output.append(f"**UID:** `{event.get('uid', 'N/A')}`")

        if event.get("start"):
            output.append(f"**Start:** {event['start']}")
        if event.get("end"):
            output.append(f"**End:** {event['end']}")
        if event.get("location"):
            output.append(f"**Location:** {event['location']}")
        if event.get("description"):
            output.append(f"**Description:** {event['description']}")
        output.append("")  # Blank line between events

    return "\n".join(output)


def truncate_response(content: str, metadata: Dict[str, Any]) -> str:
    """Truncate response if it exceeds character limit."""
    if len(content) <= CHARACTER_LIMIT:
        return content

    # Calculate truncation point
    truncation_point = CHARACTER_LIMIT - TRUNCATION_MESSAGE_RESERVE

    truncated_content = content[:truncation_point]
    truncation_message = (
        f"\n\n⚠️ **Response Truncated**\n"
        f"Original length: {len(content)} characters\n"
        f"Truncated to: {truncation_point} characters\n"
        f"Showing {metadata.get('shown_count', 'some')} of {metadata.get('total_count', 'many')} items.\n"
        f"Use date filters or limit parameter to reduce results."
    )

    return truncated_content + truncation_message


def require_read_permission(calendar_name: str) -> Optional[str]:
    """Check read permission and return error message if denied.

    Args:
        calendar_name: Name of the calendar to check

    Returns:
        Error message string if permission denied, None if allowed
    """
    if not permissions_manager.can_read(calendar_name):
        return (
            f"❌ **Permission Denied**\n\n"
            f"You don't have read permission for calendar '{calendar_name}'.\n"
            f"Use `icloud_set_calendar_permissions` to grant read access."
        )
    return None


def require_write_permission(calendar_name: str) -> Optional[str]:
    """Check write permission and return error message if denied.

    Args:
        calendar_name: Name of the calendar to check

    Returns:
        Error message string if permission denied, None if allowed
    """
    if not permissions_manager.can_write(calendar_name):
        return (
            f"❌ **Permission Denied**\n\n"
            f"You don't have write permission for calendar '{calendar_name}'.\n"
            f"Use `icloud_set_calendar_permissions` to grant write access."
        )
    return None


def get_calendar_by_name(client: caldav.DAVClient, calendar_name: str):
    """Get calendar by name from CalDAV client.

    Args:
        client: Authenticated CalDAV client
        calendar_name: Name of the calendar to fetch

    Returns:
        Calendar object

    Raises:
        NotFoundError: If calendar doesn't exist
    """
    principal = client.principal()
    return principal.calendar(name=calendar_name)


def find_event_by_uid(calendar: Any, event_uid: str) -> Optional[tuple[Any, Dict[str, Any]]]:
    """Find an event by UID in the given calendar.

    Args:
        calendar: CalDAV calendar object
        event_uid: Unique identifier of the event

    Returns:
        Tuple of (event object, event dict) if found, None otherwise
    """
    events = calendar.events()
    for event in events:
        event_dict = event_to_dict(event)
        if event_dict.get("uid") == event_uid:
            return event, event_dict
    return None


def parse_date_range(
    start_date: Optional[str],
    end_date: Optional[str],
    default_days: int = DEFAULT_DATE_RANGE_DAYS
) -> tuple[datetime, datetime]:
    """Parse date range with sensible defaults.

    Args:
        start_date: Start date in ISO format (defaults to now)
        end_date: End date in ISO format (defaults to start + default_days)
        default_days: Number of days to add to start if end not specified

    Returns:
        Tuple of (start_datetime, end_datetime)
    """
    start = parse_datetime(start_date) if start_date else datetime.now()
    end = parse_datetime(end_date) if end_date else start + timedelta(days=default_days)
    return start, end


def format_error(error_type: str, message: str, details: Optional[str] = None) -> str:
    """Format error message consistently.

    Args:
        error_type: Type of error (e.g., "Calendar Not Found", "Permission Denied")
        message: Main error message
        details: Optional additional details

    Returns:
        Formatted error string
    """
    result = f"❌ **{error_type}**\n\n{message}"
    if details:
        result += f"\n{details}"
    return result


def format_success(title: str, details: Dict[str, Any], prefix: str = "✅") -> str:
    """Format success message consistently.

    Args:
        title: Success message title
        details: Dictionary of key-value pairs to display
        prefix: Emoji/symbol prefix (default: ✅)

    Returns:
        Formatted success string
    """
    lines = [f"{prefix} **{title}**\n"]
    for key, value in details.items():
        if value is not None:
            lines.append(f"**{key}:** {value}")
    return "\n".join(lines)


# ============================================================================
# MCP Tools
# ============================================================================


@mcp.tool(
    name="icloud_list_calendars",
    annotations={
        "title": "List iCloud Calendars",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True,
    },
)
async def list_calendars(params: ListCalendarsInput) -> str:
    """List all available iCloud calendars with their permissions.

    This tool retrieves all calendars from your iCloud account and shows
    their current read/write permissions. Use this to discover which calendars
    are available and check their permission status.

    Note: This tool automatically filters out Reminders/Tasks calendars (VTODO)
    and only shows event calendars (VEVENT).

    Args:
        params (ListCalendarsInput): Input parameters containing:
            - response_format (ResponseFormat): Output format ('markdown' or 'json')

    Returns:
        str: List of calendars with permissions in the specified format

    Example Response (Markdown):
        ## iCloud Calendars

        ### Work
        - **Permissions:** Read ✓, Write ✓
        - **URL:** https://p12-caldav.icloud.com/.../calendars/work/

        ### Personal
        - **Permissions:** Read ✓, Write ✗
        - **URL:** https://p12-caldav.icloud.com/.../calendars/personal/
    """
    try:
        client = get_caldav_client()
        principal = client.principal()
        all_calendars = principal.calendars()

        # Filter out Reminders/Tasks calendars (VTODO) and keep only event calendars (VEVENT)
        calendars = [cal for cal in all_calendars if is_event_calendar(cal)]

        calendar_list = []
        for cal in calendars:
            cal_name = cal.name or "Unnamed Calendar"
            perms = permissions_manager.get_calendar_permissions(cal_name)

            calendar_list.append(
                {
                    "name": cal_name,
                    "url": str(cal.url),
                    "permissions": {
                        "read": perms.get("read", True),
                        "write": perms.get("write", False),
                    },
                }
            )

        if params.response_format == ResponseFormat.JSON:
            return json.dumps(
                {"calendars": calendar_list, "total_count": len(calendar_list)},
                indent=2,
            )
        else:
            output = ["## iCloud Calendars\n"]
            for cal in calendar_list:
                read_icon = "✓" if cal["permissions"]["read"] else "✗"
                write_icon = "✓" if cal["permissions"]["write"] else "✗"
                output.append(f"### {cal['name']}")
                output.append(
                    f"- **Permissions:** Read {read_icon}, Write {write_icon}"
                )
                output.append(f"- **URL:** `{cal['url']}`")
                output.append("")

            return "\n".join(output)

    except AuthorizationError:
        return format_error(
            "Authentication Failed",
            "Your iCloud credentials are invalid. Please check:",
            "1. ICLOUD_USERNAME is your Apple ID\n"
            "2. ICLOUD_PASSWORD is an app-specific password (not your main password)\n"
            "3. Generate app-specific passwords at: https://appleid.apple.com"
        )
    except Exception as e:
        return format_error("Error", f"Failed to list calendars: {str(e)}")


@mcp.tool(
    name="icloud_get_events",
    annotations={
        "title": "Get Calendar Events",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True,
    },
)
async def get_events(params: GetEventsInput) -> str:
    """Get events from a specific iCloud calendar.

    Retrieves events from the specified calendar within the given date range.
    Requires read permission for the calendar.

    Args:
        params (GetEventsInput): Input parameters containing:
            - calendar_name (str): Name of the calendar
            - start_date (Optional[str]): Start date (defaults to today)
            - end_date (Optional[str]): End date (defaults to 30 days from start)
            - limit (Optional[int]): Maximum number of events (default: 50, max: 250)
            - response_format (ResponseFormat): Output format ('markdown' or 'json')

    Returns:
        str: List of events in the specified format

    Example Response (Markdown):
        ### Team Meeting
        **UID:** `abc123-def456`
        **Start:** 2025-01-15 10:00:00 UTC
        **End:** 2025-01-15 11:00:00 UTC
        **Location:** Conference Room A
        **Description:** Weekly team sync
    """
    try:
        # Check read permission
        perm_error = require_read_permission(params.calendar_name)
        if perm_error:
            return perm_error

        # Parse dates
        start, end = parse_date_range(params.start_date, params.end_date)

        # Get calendar
        client = get_caldav_client()
        calendar = get_calendar_by_name(client, params.calendar_name)

        # Search for events
        events = calendar.search(start=start, end=end, event=True, expand=True)

        # Convert to dicts
        event_dicts = [event_to_dict(event) for event in events]
        event_dicts = event_dicts[: params.limit]  # Apply limit

        if params.response_format == ResponseFormat.JSON:
            result = json.dumps(
                {
                    "events": event_dicts,
                    "calendar": params.calendar_name,
                    "count": len(event_dicts),
                    "start_date": start.isoformat(),
                    "end_date": end.isoformat(),
                },
                indent=2,
            )
        else:
            header = f"## Events from '{params.calendar_name}'\n"
            header += f"**Date Range:** {format_datetime(start)} to {format_datetime(end)}\n"
            header += f"**Count:** {len(event_dicts)} events\n\n"
            result = header + format_events_markdown(event_dicts)

        # Check and truncate if needed
        if len(result) > CHARACTER_LIMIT:
            return truncate_response(
                result,
                {
                    "shown_count": len(event_dicts),
                    "total_count": len(events),
                },
            )

        return result

    except NotFoundError:
        return format_error(
            "Calendar Not Found",
            f"Calendar '{params.calendar_name}' does not exist.",
            "Use `icloud_list_calendars` to see available calendars."
        )
    except Exception as e:
        return format_error("Error", f"Failed to get events: {str(e)}")


@mcp.tool(
    name="icloud_search_events",
    annotations={
        "title": "Search Events Across Calendars",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True,
    },
)
async def search_events(params: SearchEventsInput) -> str:
    """Search for events across multiple calendars.

    Searches for events matching the query string in their title or description.
    Only searches calendars with read permission.

    Args:
        params (SearchEventsInput): Input parameters containing:
            - query (str): Search query to match against event titles/descriptions
            - start_date (Optional[str]): Start date (defaults to today)
            - end_date (Optional[str]): End date (defaults to 30 days from start)
            - calendar_names (Optional[List[str]]): Specific calendars to search (defaults to all readable)
            - limit (Optional[int]): Maximum number of events (default: 50, max: 250)
            - response_format (ResponseFormat): Output format ('markdown' or 'json')

    Returns:
        str: Search results in the specified format
    """
    try:
        # Parse dates
        start, end = parse_date_range(params.start_date, params.end_date)

        # Get calendars to search
        client = get_caldav_client()
        principal = client.principal()
        all_calendars = principal.calendars()

        if params.calendar_names:
            calendars_to_search = [
                cal
                for cal in all_calendars
                if cal.name in params.calendar_names
                and permissions_manager.can_read(cal.name)
            ]
        else:
            calendars_to_search = [
                cal for cal in all_calendars if permissions_manager.can_read(cal.name)
            ]

        # Search across calendars
        all_events = []
        query_lower = params.query.lower()

        for calendar in calendars_to_search:
            try:
                events = calendar.search(start=start, end=end, event=True, expand=True)
                for event in events:
                    event_dict = event_to_dict(event)
                    title_match = query_lower in event_dict.get("title", "").lower()
                    desc_match = query_lower in event_dict.get(
                        "description", ""
                    ).lower()

                    if title_match or desc_match:
                        event_dict["calendar"] = calendar.name
                        all_events.append(event_dict)

                    if len(all_events) >= params.limit:
                        break
            except Exception:
                continue  # Skip calendars that error

            if len(all_events) >= params.limit:
                break

        # Apply limit
        all_events = all_events[: params.limit]

        if params.response_format == ResponseFormat.JSON:
            result = json.dumps(
                {
                    "events": all_events,
                    "query": params.query,
                    "count": len(all_events),
                    "calendars_searched": [cal.name for cal in calendars_to_search],
                },
                indent=2,
            )
        else:
            header = f"## Search Results for '{params.query}'\n"
            header += f"**Calendars Searched:** {', '.join(cal.name for cal in calendars_to_search)}\n"
            header += f"**Count:** {len(all_events)} events\n\n"

            # Format with calendar info
            event_output = []
            for event in all_events:
                event_output.append(f"### {event.get('title', 'Untitled Event')}")
                event_output.append(f"**Calendar:** {event.get('calendar', 'Unknown')}")
                event_output.append(f"**UID:** `{event.get('uid', 'N/A')}`")
                if event.get("start"):
                    event_output.append(f"**Start:** {event['start']}")
                if event.get("end"):
                    event_output.append(f"**End:** {event['end']}")
                if event.get("location"):
                    event_output.append(f"**Location:** {event['location']}")
                if event.get("description"):
                    event_output.append(f"**Description:** {event['description']}")
                event_output.append("")

            result = header + "\n".join(event_output)

        return result

    except Exception as e:
        return format_error("Error", f"Failed to search events: {str(e)}")


@mcp.tool(
    name="icloud_create_event",
    annotations={
        "title": "Create Calendar Event",
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": False,
        "openWorldHint": True,
    },
)
async def create_event(params: CreateEventInput) -> str:
    """Create a new event in the specified iCloud calendar.

    Creates a calendar event with the specified details. Requires write
    permission for the target calendar.

    Args:
        params (CreateEventInput): Input parameters containing:
            - calendar_name (str): Name of the calendar
            - title (str): Event title/summary
            - start (str): Start datetime in ISO format
            - end (str): End datetime in ISO format
            - description (Optional[str]): Event description
            - location (Optional[str]): Event location
            - all_day (bool): Whether this is an all-day event

    Returns:
        str: Success message with event UID or error message

    Example:
        ✅ **Event Created Successfully**
        
        **Title:** Team Meeting
        **Calendar:** Work
        **UID:** `abc123-def456`
        **Start:** 2025-01-15 10:00:00 UTC
        **End:** 2025-01-15 11:00:00 UTC
    """
    try:
        # Check write permission
        perm_error = require_write_permission(params.calendar_name)
        if perm_error:
            return perm_error

        # Get calendar
        client = get_caldav_client()
        calendar = get_calendar_by_name(client, params.calendar_name)

        # Parse dates
        start_dt = parse_datetime(params.start)
        end_dt = parse_datetime(params.end)

        # Create event using caldav's save_event method
        event_params = {
            "summary": params.title,
            "dtstart": start_dt,
            "dtend": end_dt,
        }

        if params.description:
            event_params["description"] = params.description
        if params.location:
            event_params["location"] = params.location

        # Create the event
        created_event = calendar.save_event(**event_params)

        # Get the UID from the created event
        event_dict = event_to_dict(created_event)

        return format_success(
            "Event Created Successfully",
            {
                "Title": params.title,
                "Calendar": params.calendar_name,
                "UID": f"`{event_dict.get('uid', 'unknown')}`",
                "Start": format_datetime(start_dt),
                "End": format_datetime(end_dt),
                "Location": params.location if params.location else None,
                "Description": params.description if params.description else None,
            }
        )

    except NotFoundError:
        return format_error(
            "Calendar Not Found",
            f"Calendar '{params.calendar_name}' does not exist.",
            "Use `icloud_list_calendars` to see available calendars."
        )
    except Exception as e:
        return format_error("Error", f"Failed to create event: {str(e)}")


@mcp.tool(
    name="icloud_update_event",
    annotations={
        "title": "Update Calendar Event",
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True,
    },
)
async def update_event(params: UpdateEventInput) -> str:
    """Update an existing event in the specified iCloud calendar.

    Updates event properties. Only specified fields will be updated.
    Requires write permission for the calendar.

    Args:
        params (UpdateEventInput): Input parameters containing:
            - calendar_name (str): Name of the calendar
            - event_uid (str): Unique identifier of the event to update
            - title (Optional[str]): New event title
            - start (Optional[str]): New start datetime
            - end (Optional[str]): New end datetime
            - description (Optional[str]): New description
            - location (Optional[str]): New location

    Returns:
        str: Success message or error message
    """
    try:
        # Check write permission
        perm_error = require_write_permission(params.calendar_name)
        if perm_error:
            return perm_error

        # Get calendar
        client = get_caldav_client()
        calendar = get_calendar_by_name(client, params.calendar_name)

        # Find event by UID
        result = find_event_by_uid(calendar, params.event_uid)
        if not result:
            return format_error(
                "Event Not Found",
                f"No event with UID '{params.event_uid}' found in calendar '{params.calendar_name}'."
            )

        target_event, _ = result

        # Parse and modify event
        cal = iCalendar.from_ical(target_event.data)
        for component in cal.walk():
            if component.name == "VEVENT":
                if params.title:
                    component["summary"] = params.title
                if params.start:
                    component["dtstart"].dt = parse_datetime(params.start)
                if params.end:
                    component["dtend"].dt = parse_datetime(params.end)
                if params.description is not None:
                    component["description"] = params.description
                if params.location is not None:
                    component["location"] = params.location

                # Update last-modified timestamp
                component["last-modified"] = datetime.now(timezone.utc)
                break  # Only update the first VEVENT

        # Save updated event
        target_event.data = cal.to_ical().decode("utf-8")
        target_event.save()

        return format_success(
            "Event Updated Successfully",
            {
                "Calendar": params.calendar_name,
                "UID": f"`{params.event_uid}`",
                "New Title": params.title if params.title else None,
                "New Start": format_datetime(parse_datetime(params.start)) if params.start else None,
                "New End": format_datetime(parse_datetime(params.end)) if params.end else None,
            }
        )

    except NotFoundError:
        return format_error(
            "Calendar Not Found",
            f"Calendar '{params.calendar_name}' does not exist."
        )
    except Exception as e:
        return format_error("Error", f"Failed to update event: {str(e)}")


@mcp.tool(
    name="icloud_delete_event",
    annotations={
        "title": "Delete Calendar Event",
        "readOnlyHint": False,
        "destructiveHint": True,
        "idempotentHint": True,
        "openWorldHint": True,
    },
)
async def delete_event(params: DeleteEventInput) -> str:
    """Delete an event from the specified iCloud calendar.

    Permanently deletes the event. This action cannot be undone.
    Requires write permission for the calendar.

    Args:
        params (DeleteEventInput): Input parameters containing:
            - calendar_name (str): Name of the calendar
            - event_uid (str): Unique identifier of the event to delete

    Returns:
        str: Success message or error message
    """
    try:
        # Check write permission
        perm_error = require_write_permission(params.calendar_name)
        if perm_error:
            return perm_error

        # Get calendar
        client = get_caldav_client()
        calendar = get_calendar_by_name(client, params.calendar_name)

        # Find event by UID
        result = find_event_by_uid(calendar, params.event_uid)
        if not result:
            return format_error(
                "Event Not Found",
                f"No event with UID '{params.event_uid}' found in calendar '{params.calendar_name}'."
            )

        target_event, event_dict = result
        event_title = event_dict.get("title", "Unknown")

        # Delete the event
        target_event.delete()

        success_msg = format_success(
            "Event Deleted Successfully",
            {
                "Title": event_title,
                "Calendar": params.calendar_name,
                "UID": f"`{params.event_uid}`",
            }
        )
        return success_msg + "\n\n⚠️ This action cannot be undone."

    except NotFoundError:
        return format_error(
            "Calendar Not Found",
            f"Calendar '{params.calendar_name}' does not exist."
        )
    except Exception as e:
        return format_error("Error", f"Failed to delete event: {str(e)}")


@mcp.tool(
    name="icloud_get_calendar_permissions",
    annotations={
        "title": "Get Calendar Permissions",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    },
)
async def get_calendar_permissions() -> str:
    """Get the current permission settings for all calendars.

    Shows read and write permissions for each configured calendar.
    Calendars not explicitly configured default to read-only.

    Returns:
        str: Current permission settings in markdown format

    Example Response:
        ## Calendar Permissions

        ### Work
        - **Read:** ✓ Allowed
        - **Write:** ✓ Allowed

        ### Personal
        - **Read:** ✓ Allowed
        - **Write:** ✗ Denied
    """
    try:
        perms = permissions_manager.get_all_permissions()

        if not perms:
            return (
                "## Calendar Permissions\n\n"
                "No permissions configured yet. All calendars default to read-only.\n\n"
                "Use `icloud_set_calendar_permissions` to configure access."
            )

        output = ["## Calendar Permissions\n"]
        for cal_name, cal_perms in perms.items():
            read_status = "✓ Allowed" if cal_perms.get("read", True) else "✗ Denied"
            write_status = (
                "✓ Allowed" if cal_perms.get("write", False) else "✗ Denied"
            )

            output.append(f"### {cal_name}")
            output.append(f"- **Read:** {read_status}")
            output.append(f"- **Write:** {write_status}")
            output.append("")

        return "\n".join(output)

    except Exception as e:
        return format_error("Error", f"Failed to get permissions: {str(e)}")


@mcp.tool(
    name="icloud_set_calendar_permissions",
    annotations={
        "title": "Set Calendar Permissions",
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    },
)
async def set_calendar_permissions(params: SetPermissionsInput) -> str:
    """Set read and write permissions for a specific calendar.

    Configure whether Claude can read from and/or write to a calendar.
    Write permission includes creating, updating, and deleting events.

    Args:
        params (SetPermissionsInput): Input parameters containing:
            - calendar_name (str): Name of the calendar to configure
            - read (bool): Allow read access (default: True)
            - write (bool): Allow write access (default: False)

    Returns:
        str: Confirmation message

    Example:
        ✅ **Permissions Updated**
        
        Calendar: Work
        - Read: ✓ Allowed
        - Write: ✓ Allowed
    """
    try:
        permissions_manager.set_calendar_permissions(
            params.calendar_name, read=params.read, write=params.write
        )

        read_status = "✓ Allowed" if params.read else "✗ Denied"
        write_status = "✓ Allowed" if params.write else "✗ Denied"

        return (
            f"✅ **Permissions Updated**\n\n"
            f"**Calendar:** {params.calendar_name}\n"
            f"- **Read:** {read_status}\n"
            f"- **Write:** {write_status}\n\n"
            f"Permissions saved to: `{PERMISSIONS_FILE}`"
        )

    except Exception as e:
        return format_error("Error", f"Failed to set permissions: {str(e)}")


# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    # Check for credentials
    username = os.environ.get("ICLOUD_USERNAME")
    password = os.environ.get("ICLOUD_PASSWORD")

    if not username or not password:
        print("\n" + "=" * 70)
        print("⚠️  iCloud Calendar MCP Server - Configuration Required")
        print("=" * 70)
        print("\nMissing required environment variables:")
        print("  - ICLOUD_USERNAME: Your Apple ID (e.g., user@icloud.com)")
        print("  - ICLOUD_PASSWORD: Your app-specific password")
        print("\nTo generate an app-specific password:")
        print("  1. Go to https://appleid.apple.com")
        print("  2. Sign in with your Apple ID")
        print("  3. Navigate to Security > App-Specific Passwords")
        print("  4. Click 'Generate Password'")
        print("  5. Copy the 16-character password")
        print("\nSet environment variables:")
        print('  export ICLOUD_USERNAME="your@email.com"')
        print('  export ICLOUD_PASSWORD="xxxx-xxxx-xxxx-xxxx"')
        print("\nThen run the server again.")
        print("=" * 70 + "\n")
        exit(1)

    print("\n" + "=" * 70)
    print("✅ iCloud Calendar MCP Server Started")
    print("=" * 70)
    print(f"\nConnecting to: {CALDAV_URL}")
    print(f"Username: {username}")
    print(f"Permissions file: {PERMISSIONS_FILE}")
    print("\nAvailable tools:")
    print("  - icloud_list_calendars")
    print("  - icloud_get_events")
    print("  - icloud_search_events")
    print("  - icloud_create_event")
    print("  - icloud_update_event")
    print("  - icloud_delete_event")
    print("  - icloud_get_calendar_permissions")
    print("  - icloud_set_calendar_permissions")
    print("\nServer is ready for connections...")
    print("=" * 70 + "\n")

    # Run the MCP server
    mcp.run()