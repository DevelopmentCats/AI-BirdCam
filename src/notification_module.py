"""
Notification Module for Bird Classification System

This module provides functionality to send notifications with bird classification results
to mobile devices (via Pushbullet) and Discord (via webhooks).
"""

import os
import time
import logging
from datetime import datetime
from typing import Dict, Optional, Union, Any

# For Discord notifications
from discord_webhook import DiscordWebhook, DiscordEmbed

# For Pushbullet notifications
from pushbullet import Pushbullet

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("BirdNotifier")

class NotificationError(Exception):
    """Base exception for notification errors."""
    pass

class DiscordNotificationError(NotificationError):
    """Exception raised for Discord notification errors."""
    pass

class MobileNotificationError(NotificationError):
    """Exception raised for mobile notification errors."""
    pass

class ConfigurationError(NotificationError):
    """Exception raised for configuration errors."""
    pass

class BirdNotifier:
    """
    A class to handle notifications for bird classification results.
    Supports Discord webhooks and Pushbullet for mobile notifications.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the notifier with configuration settings.
        
        Args:
            config: Dictionary containing configuration settings
                - discord_enabled: Whether Discord notifications are enabled
                - discord_webhook_url: URL for Discord webhook
                - mobile_enabled: Whether mobile notifications are enabled
                - pushbullet_api_key: API key for Pushbullet
        """
        self.config = config
        
        # Initialize notification services based on configuration
        self.discord_enabled = config.get('discord_enabled', False)
        self.mobile_enabled = config.get('mobile_enabled', False)
        self.pb = None
        
        # Validate configuration if services are enabled
        if self.discord_enabled:
            self._validate_discord_config()
            
        if self.mobile_enabled:
            self._validate_mobile_config()
            self._initialize_pushbullet()
    
    def _validate_discord_config(self) -> None:
        """Validate Discord configuration settings."""
        if not self.config.get('discord_webhook_url'):
            raise ConfigurationError("Discord webhook URL is required when Discord notifications are enabled")
    
    def _validate_mobile_config(self) -> None:
        """Validate mobile notification configuration settings."""
        if not self.config.get('pushbullet_api_key'):
            raise ConfigurationError("Pushbullet API key is required when mobile notifications are enabled")
    
    def _initialize_pushbullet(self) -> None:
        """Initialize Pushbullet client."""
        try:
            self.pb = Pushbullet(self.config['pushbullet_api_key'])
            logger.info("Pushbullet initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Pushbullet: {e}")
            self.mobile_enabled = False
            self.pb = None
    
    def notify(self, bird_data: Dict[str, Any], image_path: str) -> Dict[str, bool]:
        """
        Send notifications with bird classification results.
        
        Args:
            bird_data: Dictionary containing bird classification results
                - species: Name of the bird species
                - confidence: Confidence score (0-100)
                - timestamp: Time of classification (optional)
            image_path: Path to the captured bird image
            
        Returns:
            Dictionary with status of each notification method
        """
        timestamp = bird_data.get('timestamp', datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        results = {
            'discord': False,
            'mobile': False
        }
        
        # Send Discord notification if enabled
        if self.discord_enabled:
            try:
                self._send_discord_notification(bird_data, image_path, timestamp)
                results['discord'] = True
                logger.info("Discord notification sent successfully")
            except Exception as e:
                logger.error(f"Failed to send Discord notification: {e}")
        
        # Send mobile notification if enabled
        if self.mobile_enabled and self.pb is not None:
            try:
                self._send_mobile_notification(bird_data, image_path, timestamp)
                results['mobile'] = True
                logger.info("Mobile notification sent successfully")
            except Exception as e:
                logger.error(f"Failed to send mobile notification: {e}")
        
        return results
    
    def _send_discord_notification(self, bird_data: Dict[str, Any], image_path: str, timestamp: str) -> None:
        """
        Send notification to Discord via webhook.
        
        Args:
            bird_data: Dictionary containing bird classification results
            image_path: Path to the captured bird image
            timestamp: Time of classification
            
        Raises:
            DiscordNotificationError: If the notification fails
        """
        try:
            webhook_url = self.config['discord_webhook_url']
            webhook = DiscordWebhook(url=webhook_url)
            
            # Create embed for better formatting
            embed = DiscordEmbed(
                title="Bird Classification Result",
                description=f"A bird has been identified!",
                color="03b2f8"  # Blue color
            )
            
            # Add fields with bird information
            embed.add_embed_field(name="Species", value=bird_data['species'])
            embed.add_embed_field(name="Confidence", value=f"{bird_data['confidence']:.2f}%")
            embed.add_embed_field(name="Timestamp", value=timestamp)
            
            # Set footer
            embed.set_footer(text="Bird Classification System")
            
            # Add timestamp
            embed.set_timestamp()
            
            # Add the embed to webhook
            webhook.add_embed(embed)
            
            # Add image file
            with open(image_path, "rb") as f:
                webhook.add_file(file=f.read(), filename=os.path.basename(image_path))
            
            # Execute the webhook
            response = webhook.execute()
            
            if not response:
                raise DiscordNotificationError("Failed to send Discord notification: No response")
                
        except Exception as e:
            raise DiscordNotificationError(f"Failed to send Discord notification: {str(e)}")
    
    def _send_mobile_notification(self, bird_data: Dict[str, Any], image_path: str, timestamp: str) -> None:
        """
        Send notification to mobile device via Pushbullet.
        
        Args:
            bird_data: Dictionary containing bird classification results
            image_path: Path to the captured bird image
            timestamp: Time of classification
            
        Raises:
            MobileNotificationError: If the notification fails
        """
        if not self.pb:
            raise MobileNotificationError("Pushbullet is not initialized")
        
        try:
            # Create notification title and body
            title = f"Bird Detected: {bird_data['species']}"
            body = f"Confidence: {bird_data['confidence']:.2f}%\nTimestamp: {timestamp}"
            
            # Send notification with file
            with open(image_path, "rb") as image_file:
                file_data = self.pb.upload_file(image_file, os.path.basename(image_path))
                
            # Push the file with the notification
            push = self.pb.push_file(**file_data, body=body, title=title)
            
            if not push:
                raise MobileNotificationError("Failed to send mobile notification: No response")
                
        except Exception as e:
            raise MobileNotificationError(f"Failed to send mobile notification: {str(e)}")