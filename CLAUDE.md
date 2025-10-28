# CLAUDE.md

## Project Overview

This repository contains an iCloud Calendar MCP (Model Context Protocol) server that enables LLMs to interact with Apple iCloud Calendars via the CalDAV protocol. The server implements granular per-calendar permission management for secure read/write access control.

**Key Technologies:**
- Python 3 with async/await patterns
- FastMCP framework for MCP server implementation
- CalDAV protocol via `python-caldav` library
- Pydantic for input validation and data models
- iCalendar (RFC 5545) format for event data

## Development Commands

### Running the Server
```bash
# Run the MCP server directly
python3 icloud_calendar_mcp.py

# Interactive setup wizard
python3 setup.py
```

### Testing and Dependencies
```bash
# Install dependencies
pip install -r requirements.txt

# Test connection manually (requires ICLOUD_USERNAME and ICLOUD_PASSWORD env vars)
python3 -c "from icloud_calendar_mcp import get_caldav_client; get_caldav_client()"
```

### Environment Configuration

The server requires two environment variables:
- `ICLOUD_USERNAME`: Apple ID (iCloud email)
- `ICLOUD_PASSWORD`: App-specific password (generate at appleid.apple.com)

**For Development (Recommended):**
```bash
# 1. Copy the example file
cp .env.example .env

# 2. Edit .env with your credentials
# ICLOUD_USERNAME=your@email.com
# ICLOUD_PASSWORD=xxxx-xxxx-xxxx-xxxx

# 3. Run the server (automatically loads .env)
python3 icloud_calendar_mcp.py
```

**For MCP Clients (example: Claude Desktop):**
Add credentials to MCP client configuration (e.g., `~/Library/Application Support/Claude/claude_desktop_config.json`):
```json
{
  "mcpServers": {
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

**Note:** Both `icloud_calendar_mcp.py` and `setup.py` automatically load from `.env` using `python-dotenv`.

## Architecture

### Core Components

**`icloud_calendar_mcp.py`** - Main server implementation with five architectural layers:

1. **Performance Optimization Layer** (lines 64-416)
   - **CalDAV Async Wrapper** (`run_caldav_async()`, lines 68-90): Wraps synchronous CalDAV operations with `asyncio.to_thread()` for concurrent execution via thread pools. Integrates with rate limiter.
   - **CalDAV Client Pool** (`CalDAVClientPool`, lines 93-156): Connection pooling with single reusable client instance, 30-minute TTL before re-authentication. Thread-safe with `Lock`. Eliminates repeated authentication overhead.
   - **Calendar Cache** (`CalendarCache`, lines 158-208): LRU cache for calendar object lookups, 5-minute TTL. Keyed by `(client_id, calendar_name)` tuple. Thread-safe with `Lock`.
   - **Event UID Cache** (`EventUIDCache`, lines 210-292): LRU cache for event lookups during update/delete operations, 1000 entry max with 60-second TTL. Per-calendar invalidation after write operations. Thread-safe with `Lock`.
   - **Rate Limiter** (`RateLimiter`, lines 295-335): Token bucket algorithm at 5 requests/second default. Async-friendly with sleep-based waiting. Thread-safe with `Lock`.
   - **DateTime Format Cache** (`DateTimeFormatCache`, lines 338-408): Pattern-based caching for datetime parsing strategies, 100 pattern max. Evicts half when full.

2. **Calendar Filtering Layer** (`is_event_calendar()` function, lines 391-449)
   - Automatically filters out Reminders/Tasks calendars (VTODO components)
   - Only processes event calendars (VEVENT components)
   - Uses CalDAV `supported-calendar-component-set` property when available
   - Fallback: samples calendar items to detect component types
   - Conservative error handling: includes calendars on errors to avoid false negatives
   - Applied in `icloud_list_calendars()` and `event_to_dict()`

3. **Permissions Layer** (`CalendarPermissions` class, lines 52-98)
   - Manages per-calendar read/write permissions
   - Stores permissions in `~/.icloud_calendar_permissions.json`
   - Default: read-only access for all calendars
   - All write operations validate permissions before execution

4. **Input Validation Layer** (Pydantic models, lines 110-308)
   - `ListCalendarsInput`, `GetEventsInput`, `SearchEventsInput`
   - `CreateEventInput`, `UpdateEventInput`, `DeleteEventInput`
   - `SetPermissionsInput`
   - Validates all tool inputs with type checking and constraints

5. **MCP Tools Layer** (8 async tool functions, lines 500+)
   - `@mcp.tool()` decorated async functions
   - Each tool performs permission checks before CalDAV operations
   - Returns formatted responses (Markdown or JSON)
   - Character limit of 25,000 to prevent token overflow
   - Uses `asyncio.gather()` for concurrent multi-calendar operations

**`setup.py`** - Interactive setup wizard that:
- Validates dependencies and credentials
- Tests CalDAV connection to iCloud
- Discovers available calendars
- Configures initial permissions interactively
- Generates MCP client configuration template in `mcp_config.json`

### Data Flow

```
LLM Request → MCP Tool → Pydantic Validation → Permission Check
                                                       ↓
                                            CalDAV Client Pool (30min cache)
                                                       ↓
                                            run_caldav_async() wrapper
                                                       ↓
                                            Thread Pool (asyncio.to_thread)
                                                       ↓
                                            Rate Limiter (5 req/sec)
                                                       ↓
                                            CalDAV Operations → iCloud
                                                       ↓
Multi-calendar queries → asyncio.gather() (concurrent ops)
                                                       ↓
                                            Calendar Cache (5min TTL)
                                                       ↓
                                            Filter Calendars (VEVENT only)
                                                       ↓
                                            Event UID Cache (60s TTL, for updates/deletes)
                                                       ↓
                                            Process Components (Skip VTODO)
                                                       ↓
                                            Format Response (MD/JSON)
                                                       ↓
                                            Truncate if needed (25k char limit)
                                                       ↓
                                            Return to LLM

Write operations (create/update/delete) → invalidate_calendar() on Event UID Cache
```

### Permission System

- **File Location**: `~/.icloud_calendar_permissions.json`
- **Format**: `{"Calendar Name": {"read": bool, "write": bool}}`
- **Defaults**: New calendars get `{"read": true, "write": false}`
- **Validation**: Every write operation (create/update/delete) checks permissions
- **Management**: Use `icloud_set_calendar_permissions` tool or edit JSON directly

## Key Design Patterns

### Async Operations
All MCP tools use `async def` and implement concurrent operations via:
- **Thread Pool Delegation**: `run_caldav_async()` wraps blocking CalDAV calls with `asyncio.to_thread()` for non-blocking execution
- **Parallel Calendar Processing**: Multi-calendar operations use `asyncio.gather()` to process calendars concurrently (e.g., `search_events` across multiple calendars)
- **Concurrent Filtering**: Calendar type detection (VEVENT vs VTODO) runs in parallel via `asyncio.gather([run_caldav_async(is_event_calendar, cal) for cal in calendars])`
- **Thread Safety**: All caches and pools use `threading.Lock` for thread-safe access

### Error Handling
- Try-catch blocks around all CalDAV operations
- User-friendly error messages with status indicators
- Specific error types: `AuthorizationError`, `NotFoundError`, permission denials

### Response Formatting
- **Markdown**: Human-readable format for LLM conversations
- **JSON**: Machine-readable format for programmatic access
- **Truncation**: Responses limited to 25,000 characters with metadata preserved

### DateTime Handling
- Input: ISO 8601 format (YYYY-MM-DD or YYYY-MM-DDTHH:MM:SS)
- Processing: Python `datetime` with timezone awareness
- Output: ISO format strings for consistency
- Helper functions: `parse_datetime()` and `format_datetime()`

## Important Constraints

1. **Authentication**: Must use app-specific passwords (not main Apple password) due to 2FA
2. **Rate Limiting**: Apple may rate-limit excessive CalDAV requests
3. **No OAuth**: iCloud doesn't support OAuth 2.0 for CalDAV
4. **No Webhooks**: Polling required; no push notifications available
5. **Full Updates**: CalDAV requires PUT (full object) for updates, no PATCH support
6. **Event Limits**: Default 50 events per query, max 250

## Security Considerations

- Never commit credentials to version control
- App-specific passwords provide limited scope vs. main Apple password
- Write permissions are dangerous (allow create/update/delete)
- Start with read-only access and grant write selectively
- Permissions persist across sessions in home directory

## Common Modification Patterns

### Adding a New Tool
1. Define Pydantic input model (inherit from `BaseModel`)
2. Create `@mcp.tool()` decorated async function
3. Add permission validation if needed
4. Implement CalDAV operation with error handling using `run_caldav_async()` wrapper
5. If write operation (create/update/delete), call `_event_uid_cache.invalidate_calendar(calendar_url)` after success
6. Format response (Markdown/JSON)
7. Apply `truncate_response()` if needed

### Modifying Event Schema
1. Update relevant Pydantic model (`CreateEventInput`, `UpdateEventInput`)
2. Modify `event_to_dict()` function for serialization
3. Update `format_events_markdown()` for display
4. Adjust iCalendar object creation in tool functions

### Extending Permission Model
1. Modify `CalendarPermissions` class methods
2. Update `~/.icloud_calendar_permissions.json` schema
3. Add validation in tool functions
4. Update `get_calendar_permissions` and `set_calendar_permissions` tools
