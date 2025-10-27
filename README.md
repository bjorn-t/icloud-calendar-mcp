# iCloud Calendar MCP Server

MCP server for iCloud Calendar integration via CalDAV protocol. Enables LLMs (Claude, ChatGPT, etc.) to manage calendar events with granular per-calendar permissions.

## ⚠️ Important Disclaimer

This is an **unofficial, third-party open-source tool**. It is **not affiliated with, endorsed by, or sponsored by Apple Inc.**

This tool uses the standard CalDAV protocol (RFC 4791) to access calendar data via Apple's publicly documented CalDAV endpoint. iCloud and Apple Calendar are trademarks of Apple Inc., registered in the U.S. and other countries.

## Features

- Per-calendar read/write permission controls
- List calendars, get/search/create/update/delete events
- Automatic filtering of Reminders/Tasks calendars (VTODO)
- JSON and Markdown output formats
- CalDAV protocol with app-specific password authentication
- Caching for improved performance

## Requirements

- Python 3.8+
- iCloud account with two-factor authentication
- App-specific password (not your main iCloud password)
- MCP-compatible client

## Installation

```bash
# Clone repository
git clone https://github.com/bjorn-t/icloud-calendar-mcp.git
cd icloud-calendar-mcp

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### 1. Generate App-Specific Password

1. Go to https://appleid.apple.com
2. Sign in and navigate to Security → App-Specific Passwords
3. Generate a new password
4. Save the 16-character password (format: `xxxx-xxxx-xxxx-xxxx`)

### 2. Configure Your MCP Client

Add the server to your MCP client configuration with your credentials:

**Claude Desktop** (macOS: `~/Library/Application Support/Claude/claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "icloud-calendar": {
      "command": "python3",
      "args": ["/absolute/path/to/icloud_calendar_mcp.py"],
      "env": {
        "ICLOUD_USERNAME": "your@email.com",
        "ICLOUD_PASSWORD": "xxxx-xxxx-xxxx-xxxx"
      }
    }
  }
}
```

**Other MCP Clients**:

```json
{
  "servers": {
    "icloud-calendar": {
      "command": "python3",
      "args": ["/path/to/icloud_calendar_mcp.py"],
      "env": {
        "ICLOUD_USERNAME": "your@email.com",
        "ICLOUD_PASSWORD": "xxxx-xxxx-xxxx-xxxx"
      }
    }
  }
}
```

That's it! Restart your MCP client and you're ready to use the calendar tools.

### 3. Permissions Configuration (Optional)

Permissions are stored in `~/.icloud_calendar_permissions.json`:

```json
{
  "Work": {
    "read": true,
    "write": true
  },
  "Personal": {
    "read": true,
    "write": false
  }
}
```

**Defaults**: New calendars default to read-only (`read: true, write: false`).

## Available Tools

| Tool | Description | Required Permission |
|------|-------------|---------------------|
| `icloud_list_calendars` | List all calendars with permission status | None |
| `icloud_get_calendar_permissions` | View current permission settings | None |
| `icloud_set_calendar_permissions` | Configure read/write access per calendar | None |
| `icloud_get_events` | Get events from a calendar with date filtering | Read |
| `icloud_search_events` | Search across multiple calendars | Read |
| `icloud_create_event` | Create a new calendar event | Write |
| `icloud_update_event` | Update an existing event | Write |
| `icloud_delete_event` | Delete an event (destructive) | Write |

## Tool Reference

### icloud_list_calendars

```python
icloud_list_calendars(
    response_format: str = "markdown"  # "markdown" or "json"
)
```

### icloud_get_events

```python
icloud_get_events(
    calendar_name: str,                # Required
    start_date: str = None,            # ISO format: YYYY-MM-DD
    end_date: str = None,              # ISO format: YYYY-MM-DD
    limit: int = 50,                   # Max: 250
    response_format: str = "markdown"
)
```

### icloud_search_events

```python
icloud_search_events(
    query: str,                        # Required
    start_date: str = None,
    end_date: str = None,
    calendar_names: list = None,       # Search specific calendars
    limit: int = 50,
    response_format: str = "markdown"
)
```

### icloud_create_event

```python
icloud_create_event(
    calendar_name: str,                # Required
    title: str,                        # Required
    start: str,                        # Required, ISO format
    end: str,                          # Required, ISO format
    description: str = None,
    location: str = None,
    all_day: bool = False
)
```

### icloud_update_event

```python
icloud_update_event(
    calendar_name: str,                # Required
    event_uid: str,                    # Required
    title: str = None,
    start: str = None,                 # ISO format
    end: str = None,                   # ISO format
    description: str = None,
    location: str = None
)
```

### icloud_delete_event

```python
icloud_delete_event(
    calendar_name: str,                # Required
    event_uid: str                     # Required
)
```

### icloud_set_calendar_permissions

```python
icloud_set_calendar_permissions(
    calendar_name: str,                # Required
    read: bool = True,
    write: bool = False
)
```

## Security

**Permission System**: All write operations (create/update/delete) validate permissions before execution. Permissions are stored locally in `~/.icloud_calendar_permissions.json` and persist across sessions.

**Authentication**: Uses iCloud app-specific passwords, not your main password. Credentials are stored in environment variables or MCP client configuration.

**Best Practices**:
- Start with read-only access, add write permissions selectively
- Never commit credentials to version control
- Use separate calendars for different permission levels
- Regularly review permissions with `icloud_get_calendar_permissions`

## Troubleshooting

**Authentication Failed**
- Verify you're using an app-specific password, not your main password
- Check credentials in environment variables or client config
- Generate a new app-specific password if needed

**Calendar Not Found**
- Run `icloud_list_calendars` to see available calendars
- Check exact calendar name (case-sensitive)
- Verify calendar exists in iCloud

**Permission Denied**
- Use `icloud_set_calendar_permissions` to grant access
- Check `~/.icloud_calendar_permissions.json` for current settings

**Connection Issues**
- Verify internet connectivity
- Check if iCloud services are online: https://www.apple.com/support/systemstatus/

## Technical Details

**Protocol**: CalDAV (RFC 4791)
**Server**: `https://caldav.icloud.com/`
**Format**: iCalendar (RFC 5545)
**Library**: `python-caldav`

**Limitations**:
- No OAuth support (iCloud doesn't support OAuth 2.0 for CalDAV)
- No webhooks/push notifications
- No PATCH support (full PUT required for updates)
- Rate limiting may apply for excessive requests

**Performance**:
- Response limit: 25,000 characters (truncated if exceeded)
- Default event limit: 50 per query (max: 250)
- Calendar type detection results are cached
- Pagination via `start_date`, `end_date`, `limit` parameters

## Development

### Running the Server Standalone

For testing or development, you can run the server directly without an MCP client.

**1. Create a `.env` file** (optional - only for standalone testing):

```bash
ICLOUD_USERNAME="your@email.com"
ICLOUD_PASSWORD="xxxx-xxxx-xxxx-xxxx"
```

**2. Run the server**:

```bash
# Start the MCP server
python3 icloud_calendar_mcp.py

# Or use the interactive setup wizard
python3 setup.py
```

**Note**: The `.env` file is NOT needed if you're using this server with an MCP client (Claude Desktop, etc.). Only create it for standalone development/testing.

## Contributing

Issues and pull requests welcome. Please:
- Include error messages and reproduction steps for bugs
- Specify Python version and OS
- Follow existing code style

## License

MIT License - see [LICENSE](LICENSE) file for details.

Copyright (c) 2025 Björn Thordebrand

## Trademark Notice

iCloud, Apple Calendar, and Apple are trademarks of Apple Inc., registered in the U.S. and other countries. This project is an independent implementation using the open CalDAV standard (RFC 4791) and is not affiliated with, endorsed by, or sponsored by Apple Inc.

The use of "iCloud" in this project's name is intended solely for descriptive purposes to indicate compatibility with Apple's iCloud Calendar service. No trademark infringement is intended.

## Acknowledgments

- [python-caldav](https://github.com/python-caldav/caldav) - CalDAV client library
- [Model Context Protocol](https://modelcontextprotocol.io/) - MCP specification
- [icalendar](https://github.com/collective/icalendar) - iCalendar parsing
