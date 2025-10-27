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

## Security

Write operations validate permissions before execution. Permissions persist in `~/.icloud_calendar_permissions.json`. Default: read-only for new calendars.

Uses app-specific passwords (not main iCloud password). Never commit credentials to version control.

## Technical Details

- **Protocol**: CalDAV (RFC 4791) via `python-caldav`
- **Server**: `https://caldav.icloud.com/`
- **Format**: iCalendar (RFC 5545)
- **Limitations**: No OAuth, webhooks, or PATCH support. Full PUT required for updates.
- **Performance**: 25K char response limit, 50-250 events per query, cached calendar detection

## Development

For standalone testing, create a `.env` file with credentials and run `python3 icloud_calendar_mcp.py` or `python3 setup.py`.

## Contributing

Issues and PRs welcome. Include error messages, Python version, and OS for bugs.

## License

MIT License - see [LICENSE](LICENSE) file for details.

Copyright (c) 2025 Björn Thordebrand

## Trademark Notice

iCloud and Apple Calendar are trademarks of Apple Inc. This project uses the open CalDAV standard and is not affiliated with Apple Inc.

## Acknowledgments

- [python-caldav](https://github.com/python-caldav/caldav) - CalDAV client library
- [Model Context Protocol](https://modelcontextprotocol.io/) - MCP specification
- [icalendar](https://github.com/collective/icalendar) - iCalendar parsing
