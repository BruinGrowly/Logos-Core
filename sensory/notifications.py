"""
Notifications - User Feedback System
=====================================

Displays notifications to inform users about system learning.
Uses Windows toast notifications when available, falls back to console.

Example:
    "I see you moved 'Scan_004.pdf'. I have updated my weights 
     to understand this is Personal, not Financial."
"""

import os
import sys
from typing import Optional


def notify_correction(file_name: str, wrong_concept: str, 
                      correct_folder: str) -> bool:
    """
    Display a notification about a learned correction.
    
    Args:
        file_name: Name of the file that was corrected
        wrong_concept: The concept that was incorrectly matched
        correct_folder: Where the user moved the file
        
    Returns:
        True if notification was displayed, False otherwise
    """
    title = "Logos Learning"
    message = (
        f"I see you moved '{file_name}'.\n"
        f"I have updated my weights to understand this is "
        f"'{correct_folder}', not '{wrong_concept}'."
    )
    
    # Try Windows toast notification first
    if sys.platform == 'win32':
        if _show_windows_toast(title, message):
            return True
    
    # Fallback to console
    _show_console_notification(title, message)
    return True


def notify_learning(message: str, title: str = "Logos") -> bool:
    """
    Display a general learning notification.
    
    Args:
        message: The message to display
        title: Notification title
        
    Returns:
        True if notification was displayed
    """
    if sys.platform == 'win32':
        if _show_windows_toast(title, message):
            return True
    
    _show_console_notification(title, message)
    return True


def _show_windows_toast(title: str, message: str) -> bool:
    """
    Show a Windows toast notification.
    
    Tries multiple libraries in order of preference.
    """
    # Try win10toast
    try:
        from win10toast import ToastNotifier
        toaster = ToastNotifier()
        toaster.show_toast(
            title,
            message,
            duration=5,
            threaded=True
        )
        return True
    except ImportError:
        pass
    except Exception as e:
        print(f"[Notifications] win10toast error: {e}")
    
    # Try plyer
    try:
        from plyer import notification
        notification.notify(
            title=title,
            message=message,
            app_name="Logos-Core",
            timeout=5
        )
        return True
    except ImportError:
        pass
    except Exception as e:
        print(f"[Notifications] plyer error: {e}")
    
    # Try Windows PowerShell toast (no dependencies)
    try:
        import subprocess
        # Escape quotes in message
        safe_message = message.replace('"', '`"').replace('\n', ' ')
        safe_title = title.replace('"', '`"')
        
        ps_script = f'''
        [Windows.UI.Notifications.ToastNotificationManager, Windows.UI.Notifications, ContentType = WindowsRuntime] | Out-Null
        [Windows.Data.Xml.Dom.XmlDocument, Windows.Data.Xml.Dom.XmlDocument, ContentType = WindowsRuntime] | Out-Null
        $template = "<toast><visual><binding template='ToastText02'><text id='1'>{safe_title}</text><text id='2'>{safe_message}</text></binding></visual></toast>"
        $xml = New-Object Windows.Data.Xml.Dom.XmlDocument
        $xml.LoadXml($template)
        $toast = [Windows.UI.Notifications.ToastNotification]::new($xml)
        [Windows.UI.Notifications.ToastNotificationManager]::CreateToastNotifier("Logos-Core").Show($toast)
        '''
        
        subprocess.run(
            ['powershell', '-Command', ps_script],
            capture_output=True,
            timeout=5
        )
        return True
    except Exception as e:
        print(f"[Notifications] PowerShell toast error: {e}")
    
    return False


def _show_console_notification(title: str, message: str) -> None:
    """
    Show a notification in the console.
    
    Fallback when toast notifications aren't available.
    """
    border = "=" * 60
    print(f"\n{border}")
    print(f"  📢 {title}")
    print(f"{border}")
    for line in message.split('\n'):
        print(f"  {line}")
    print(f"{border}\n")


if __name__ == "__main__":
    # Test notification
    notify_correction(
        file_name="Scan_004.pdf",
        wrong_concept="Finance",
        correct_folder="Personal"
    )
