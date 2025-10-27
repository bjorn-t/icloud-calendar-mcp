# Disclaimer

## Third-Party Status

This project, **iCloud Calendar MCP Server**, is an **unofficial, third-party open-source tool**. It is **not affiliated with, endorsed by, or sponsored by Apple Inc.** in any way.

## Technical Implementation

This tool is built using:

- **CalDAV Protocol (RFC 4791)**: An open standard for calendar data access
- **Apple's Public CalDAV Endpoint**: `https://caldav.icloud.com/`
- **App-Specific Passwords**: Apple's official authentication method for third-party applications
- **Open Source Libraries**: python-caldav, icalendar, and other publicly available tools

This implementation does not use any:
- Reverse-engineered protocols or APIs
- Private or undocumented Apple APIs
- Proprietary Apple software or code

## Trademark Notice

The following are trademarks of Apple Inc., registered in the U.S. and other countries:
- iCloud
- Apple Calendar
- Apple
- Apple ID

The use of "iCloud" in this project's name and documentation is intended solely for **descriptive purposes** to indicate compatibility with Apple's iCloud Calendar service. This constitutes **fair use** under trademark law for the purpose of describing the product's functionality and compatibility.

## No Warranty

This software is provided "as is", without warranty of any kind, express or implied, including but not limited to the warranties of merchantability, fitness for a particular purpose, and noninfringement. See the [MIT License](LICENSE) for full details.

## User Responsibilities

Users of this software are responsible for:
- Complying with Apple's [iCloud Terms of Service](https://www.apple.com/legal/internet-services/icloud/)
- Protecting their app-specific passwords and credentials
- Understanding the permissions they grant to this tool
- Using the software in accordance with all applicable laws and regulations

## Data Privacy

This tool:
- Stores credentials only in environment variables or user-provided configuration files
- Does not transmit credentials or calendar data to any third parties
- Operates entirely between the user's system and Apple's iCloud servers
- Stores permission settings locally in `~/.icloud_calendar_permissions.json`

## Contact

For questions about this disclaimer or the project, please open an issue on the GitHub repository.

---

**Last Updated**: 2025-10-27
