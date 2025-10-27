#!/usr/bin/env python3
"""
Setup Helper for iCloud Calendar MCP Server

This script helps you configure the iCloud Calendar MCP Server by:
1. Checking for required environment variables
2. Testing the connection to iCloud
3. Discovering your calendars
4. Setting up initial permissions
"""

import os
import sys
import json
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file (if it exists)
load_dotenv()


def print_header(text):
    """Print a formatted header."""
    print("\n" + "=" * 70)
    print(f"  {text}")
    print("=" * 70 + "\n")


def print_step(number, text):
    """Print a formatted step."""
    print(f"\n{'─' * 70}")
    print(f"  Step {number}: {text}")
    print('─' * 70 + "\n")


def check_dependencies():
    """Check if required Python packages are installed."""
    print_step(1, "Checking Dependencies")

    required_packages = ["caldav", "icalendar", "pydantic", "mcp"]
    missing = []

    for package in required_packages:
        try:
            __import__(package)
            print(f"✓ {package:<20} installed")
        except ImportError:
            print(f"✗ {package:<20} MISSING")
            missing.append(package)

    if missing:
        print(f"\nWarning: Missing packages: {', '.join(missing)}")
        print("\nInstall them with:")
        print("  pip install -r requirements.txt")
        return False

    print("\nAll dependencies installed!")
    return True


def check_credentials():
    """Check if iCloud credentials are configured."""
    print_step(2, "Checking iCloud Credentials")

    username = os.environ.get("ICLOUD_USERNAME")
    password = os.environ.get("ICLOUD_PASSWORD")

    if username:
        print(f"✓ ICLOUD_USERNAME: {username}")
    else:
        print("✗ ICLOUD_USERNAME: Not set")

    if password:
        print(f"✓ ICLOUD_PASSWORD: {'*' * 16} (hidden)")
    else:
        print("✗ ICLOUD_PASSWORD: Not set")

    if not username or not password:
        print("\nWarning: Missing credentials!")
        print("\nTo set them up:")
        print("  1. Go to https://appleid.apple.com")
        print("  2. Navigate to Security → App-Specific Passwords")
        print("  3. Generate a new password")
        print("  4. Set environment variables:")
        print('     export ICLOUD_USERNAME="your@email.com"')
        print('     export ICLOUD_PASSWORD="xxxx-xxxx-xxxx-xxxx"')
        return False

    print("\nCredentials configured!")
    return True


def test_connection():
    """Test connection to iCloud CalDAV server."""
    print_step(3, "Testing Connection to iCloud")

    try:
        import caldav

        username = os.environ.get("ICLOUD_USERNAME")
        password = os.environ.get("ICLOUD_PASSWORD")

        print(f"Connecting to: https://caldav.icloud.com/")
        print(f"Username: {username}")
        print("Testing authentication...")

        client = caldav.DAVClient(
            url="https://caldav.icloud.com/", username=username, password=password
        )
        principal = client.principal()

        print("\nConnection successful!")
        return principal

    except Exception as e:
        print(f"\nConnection failed: {str(e)}")
        print("\nPossible issues:")
        print("  1. Incorrect Apple ID or app-specific password")
        print("  2. Network connectivity issues")
        print("  3. iCloud services are down")
        return None


def discover_calendars(principal):
    """Discover available calendars."""
    print_step(4, "Discovering Calendars")

    try:
        calendars = principal.calendars()
        print(f"Found {len(calendars)} calendar(s):\n")

        for i, cal in enumerate(calendars, 1):
            print(f"  {i}. {cal.name}")

        print(f"\nDiscovered {len(calendars)} calendars!")
        return calendars

    except Exception as e:
        print(f"\nFailed to discover calendars: {str(e)}")
        return []


def setup_permissions(calendars):
    """Interactive permission setup."""
    print_step(5, "Setting Up Permissions")

    permissions_file = Path.home() / ".icloud_calendar_permissions.json"
    permissions = {}

    print("Let's configure permissions for each calendar.")
    print("Default: Read-only access for all calendars\n")

    for cal in calendars:
        cal_name = cal.name
        print(f"\n📅 Calendar: {cal_name}")
        print("   Current: Read ✓, Write ✗")

        grant_write = (
            input("   Grant write access? (y/N): ").strip().lower() == "y"
        )

        permissions[cal_name] = {"read": True, "write": grant_write}

        if grant_write:
            print(f"   Updated: Read ✓, Write ✓")
        else:
            print(f"   Keeping: Read ✓, Write ✗")

    # Save permissions
    try:
        permissions_file.parent.mkdir(parents=True, exist_ok=True)
        with open(permissions_file, "w") as f:
            json.dump(permissions, f, indent=2)

        print(f"\nPermissions saved to: {permissions_file}")
        return True

    except Exception as e:
        print(f"\nFailed to save permissions: {str(e)}")
        return False


def create_mcp_config():
    """Generate MCP client configuration."""
    print_step(6, "MCP Client Configuration")

    config = {
        "mcpServers": {
            "icloud-calendar": {
                "command": "python",
                "args": [str(Path.cwd() / "icloud_calendar_mcp.py")],
                "env": {
                    "ICLOUD_USERNAME": os.environ.get("ICLOUD_USERNAME", ""),
                    "ICLOUD_PASSWORD": os.environ.get("ICLOUD_PASSWORD", ""),
                },
            }
        }
    }

    config_text = json.dumps(config, indent=2)

    print("Add this to your MCP client configuration:\n")
    print("Examples:")
    print("  Claude Desktop (macOS): ~/Library/Application Support/Claude/claude_desktop_config.json")
    print("  Claude Desktop (Windows): %APPDATA%\\Claude\\claude_desktop_config.json")
    print("  Continue.dev: ~/.continue/config.json")
    print("  Generic MCP client: Check your client's documentation\n")
    print(config_text)

    # Save to file
    config_file = Path.cwd() / "mcp_config.json"
    with open(config_file, "w") as f:
        f.write(config_text)

    print(f"\nConfiguration saved to: {config_file}")
    print("Copy this into your MCP client config file.")


def main():
    """Main setup flow."""
    print_header("iCloud Calendar MCP Server - Setup")

    print("This script will help you set up the iCloud Calendar MCP Server.")
    print("Press Ctrl+C at any time to exit.\n")

    input("Press Enter to begin setup...")

    # Step 1: Check dependencies
    if not check_dependencies():
        print("\nSetup aborted. Please install dependencies and try again.")
        return 1

    # Step 2: Check credentials
    if not check_credentials():
        print("\nSetup aborted. Please configure credentials and try again.")
        return 1

    # Step 3: Test connection
    principal = test_connection()
    if not principal:
        print("\nSetup aborted. Please fix connection issues and try again.")
        return 1

    # Step 4: Discover calendars
    calendars = discover_calendars(principal)
    if not calendars:
        print("\nWarning: No calendars found. Setup incomplete.")
        return 1

    # Step 5: Setup permissions
    if not setup_permissions(calendars):
        print("\nWarning: Failed to save permissions. You can set them manually later.")

    # Step 6: Generate MCP config
    create_mcp_config()

    # Final summary
    print_header("Setup Complete!")
    print("Next steps:")
    print("  1. Add the configuration to your MCP client config file")
    print("  2. Restart your MCP client")
    print("  3. Test: Ask your LLM to 'list my iCloud calendars'")
    print("\nTo run the server manually for testing:")
    print("  python3 icloud_calendar_mcp.py")
    print("\n" + "=" * 70 + "\n")

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\nSetup cancelled by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nUnexpected error: {str(e)}")
        sys.exit(1)