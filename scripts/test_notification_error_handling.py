#!/usr/bin/env python3
"""
Unit tests for the notification module's error handling capabilities.
"""

import unittest
from unittest.mock import patch, MagicMock
import os
import sys
from notification_module import BirdNotifier, DiscordNotificationError, MobileNotificationError

class TestNotificationErrorHandling(unittest.TestCase):
    """Test cases for notification error handling."""
    
    def setUp(self):
        """Set up test data."""
        self.bird_data = {
            'species': 'Eastern Yellow Robin',
            'confidence': 95.7,
            'timestamp': '2025-04-06 21:15:30'
        }
        self.image_path = "test_data/sample_bird.jpg"
        
        # Ensure the test image exists
        if not os.path.exists(self.image_path):
            raise FileNotFoundError(f"Test image not found: {self.image_path}")
    
    def test_discord_error_handling(self):
        """Test that Discord notification errors are handled gracefully."""
        with patch('discord_webhook.DiscordWebhook.execute') as mock_execute:
            # Simulate network error
            mock_execute.side_effect = Exception("Network error")
            
            config = {
                'discord_enabled': True,
                'discord_webhook_url': 'https://discord.com/api/webhooks/fake/url',
                'mobile_enabled': False
            }
            
            notifier = BirdNotifier(config)
            results = notifier.notify(self.bird_data, self.image_path)
            
            self.assertFalse(results['discord'], "Discord notification should have failed")
    
    @patch('pushbullet.Pushbullet')
    def test_mobile_error_handling(self, mock_pushbullet_class):
        """Test that mobile notification errors are handled gracefully."""
        # Configure the mock Pushbullet instance
        mock_pb_instance = MagicMock()
        mock_pb_instance.upload_file.side_effect = Exception("API error")
        mock_pushbullet_class.return_value = mock_pb_instance
        
        config = {
            'discord_enabled': False,
            'mobile_enabled': True,
            'pushbullet_api_key': 'fake_api_key'
        }
        
        notifier = BirdNotifier(config)
        results = notifier.notify(self.bird_data, self.image_path)
        
        self.assertFalse(results['mobile'], "Mobile notification should have failed")
    
    @patch('pushbullet.Pushbullet')
    def test_independent_notification_methods(self, mock_pushbullet_class):
        """Test that one notification method failing doesn't prevent the other."""
        # Configure the mock Pushbullet instance to succeed
        mock_pb_instance = MagicMock()
        mock_pb_instance.upload_file.return_value = {'file_name': 'test.jpg', 'file_type': 'image/jpeg', 'file_url': 'test_url'}
        mock_pb_instance.push_file.return_value = {'status': 'success'}
        mock_pushbullet_class.return_value = mock_pb_instance
        
        # Configure Discord to fail
        with patch('discord_webhook.DiscordWebhook.execute') as mock_execute:
            mock_execute.side_effect = Exception("Discord API error")
            
            config = {
                'discord_enabled': True,
                'discord_webhook_url': 'https://discord.com/api/webhooks/fake/url',
                'mobile_enabled': True,
                'pushbullet_api_key': 'fake_api_key'
            }
            
            notifier = BirdNotifier(config)
            results = notifier.notify(self.bird_data, self.image_path)
            
            self.assertFalse(results['discord'], "Discord notification should have failed")
            self.assertTrue(results['mobile'], "Mobile notification should have succeeded")

if __name__ == "__main__":
    unittest.main()