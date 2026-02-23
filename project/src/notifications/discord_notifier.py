from discord_webhook import DiscordWebhook, DiscordEmbed
from pathlib import Path
import datetime
from typing import Optional, Union


class DiscordNotifier:
    """
    A class to handle Discord notifications for training processes.

    This class provides methods to send different types of notifications:
    - Training progress updates
    - Error notifications
    - Training completion notifications
    """

    def __init__(
        self,
        webhook_url: str,
        project_name: str = "BIT Thesis Project",
        project_url: str = "https://github.com/matejkrenek/BIT-thesis",
        author_icon_url: str = "https://cdn.top.gg/icons/42644ab0451ecf075ec9612ab7df3c16.png",
        avatar_name: str = "Unknown",
        silent_mode: bool = False,
    ):
        """
        Initialize the Discord notifier.

        Args:
            webhook_url: Discord webhook URL
            project_name: Name of the project for branding
            project_url: URL to the project repository
            author_icon_url: URL to author's avatar/icon
        """
        self.webhook_url = webhook_url
        self.project_name = project_name
        self.project_url = project_url
        self.author_icon_url = author_icon_url
        self.avatar_name = avatar_name
        self.silent_mode = silent_mode

    def _create_webhook(self) -> DiscordWebhook:
        """Create a new Discord webhook instance."""
        return DiscordWebhook(url=self.webhook_url)

    def _create_base_embed(
        self, title: str, description: str, color: str
    ) -> DiscordEmbed:
        """
        Create a base embed with common settings.

        Args:
            title: Embed title
            description: Embed description
            color: Embed color (hex string)

        Returns:
            DiscordEmbed: Base embed object
        """
        embed = DiscordEmbed(title=title, description=description, color=color)

        # Set author
        embed.set_author(
            name=self.avatar_name,
            url=self.project_url,
            icon_url=self.author_icon_url,
        )

        # Set timestamp
        embed.set_timestamp()

        return embed

    def _add_image_files(
        self,
        webhook: DiscordWebhook,
        embed: DiscordEmbed,
        image_path: Optional[Union[str, Path]] = None,
        thumbnail_path: Optional[Union[str, Path]] = None,
        image_filename: str = "image.png",
        thumbnail_filename: str = "thumbnail.png",
    ) -> None:
        """
        Add image files to webhook and embed.

        Args:
            webhook: Discord webhook instance
            embed: Discord embed instance
            image_path: Path to main image file
            thumbnail_path: Path to thumbnail image file
            image_filename: Filename for the main image attachment
            thumbnail_filename: Filename for the thumbnail attachment
        """
        if image_path and Path(image_path).exists():
            with open(Path(image_path), "rb") as f:
                webhook.add_file(file=f.read(), filename=image_filename)
            embed.set_image(url=f"attachment://{image_filename}")

        if thumbnail_path and Path(thumbnail_path).exists():
            with open(Path(thumbnail_path), "rb") as f:
                webhook.add_file(file=f.read(), filename=thumbnail_filename)
            embed.set_thumbnail(url=f"attachment://{thumbnail_filename}")

    def send_training_progress(
        self,
        epoch: int,
        total_epochs: int,
        current_loss: float,
        best_loss: float,
        learning_rate: float,
        batch_size: int,
        elapsed_time: str,
        estimated_finish_time: str,
        loss_curve_path: Optional[Union[str, Path]] = None,
    ) -> dict:
        """
        Send training progress notification to Discord.

        Args:
            epoch: Current epoch number
            total_epochs: Total number of epochs
            current_loss: Current training loss
            best_loss: Best loss achieved so far
            learning_rate: Current learning rate
            batch_size: Training batch size
            elapsed_time: Time elapsed since training started
            estimated_finish_time: Estimated time to completion
            loss_curve_path: Path to loss curve image

        Returns:
            dict: Response from Discord API
        """
        if self.silent_mode: return {"status": "silent_mode_enabled"}

        webhook = self._create_webhook()

        # Progress percentage and bar
        progress_percentage = (epoch / total_epochs) * 100
        progress_bar = "█" * int(progress_percentage // 4) + "░" * (
            25 - int(progress_percentage // 4)
        )

        # Create progress embed
        embed = self._create_base_embed(
            title="🚀 Training Progress Update",
            description=f"Model training is progressing... **Epoch {epoch}/{total_epochs}**",
            color="3498db",  # Nice blue for progress
        )

        # Add images
        self._add_image_files(
            webhook,
            embed,
            image_path=loss_curve_path,
            image_filename="loss_curve.png",
            thumbnail_filename="sample_output.png",
        )

        # Progress bar field
        embed.add_embed_field(
            name="Progress",
            value=f"`{progress_bar}` {progress_percentage:.1f}%",
            inline=False,
        )

        # Best loss information
        embed.add_embed_field(
            name="Current Loss",
            value=f"{current_loss:.6f}",
            inline=True,
        )

        # Best loss information
        embed.add_embed_field(
            name="Best Loss",
            value=f"{best_loss:.6f}",
            inline=True,
        )

        # Learning rate
        embed.add_embed_field(
            name="Learning Rate",
            value=f"{learning_rate}",
            inline=True,
        )

        # Batch size
        embed.add_embed_field(
            name="Batch size",
            value=f"{batch_size}",
            inline=True,
        )

        # Elapsed time
        embed.add_embed_field(
            name="Elapsed",
            value=f"{elapsed_time}",
            inline=True,
        )

        # Estimated finish time
        embed.add_embed_field(
            name="ETA",
            value=f"{estimated_finish_time}",
            inline=True,
        )

        # Set footer
        embed.set_footer(text=f"{self.project_name} - Training Progress")

        webhook.add_embed(embed)
        response = webhook.execute()
        return response

    def send_training_error(
        self,
        error_message: str,
        epoch: Optional[int] = None,
        traceback_info: Optional[str] = None,
    ) -> dict:
        """
        Send training error notification to Discord.

        Args:
            error_message: The error message that occurred
            epoch: Epoch number where error occurred (optional)
            traceback_info: Stack trace information (optional)

        Returns:
            dict: Response from Discord API
        """
        if self.silent_mode: return {"status": "silent_mode_enabled"}

        webhook = self._create_webhook()

        # Create error embed
        embed = self._create_base_embed(
            title="❌ Training Error Occurred",
            description="An error occurred during model training. Immediate attention required!",
            color="e74c3c",  # Red for error
        )

        # Error details
        embed.add_embed_field(
            name="Error Message", value=f"```{error_message[:1000]}```", inline=False
        )

        # Epoch information if available
        if epoch is not None:
            embed.add_embed_field(name="Failed at Epoch", value=f"{epoch}", inline=True)

        # Add timestamp of error
        embed.add_embed_field(
            name="Error Time",
            value=f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            inline=True,
        )

        # Traceback information if available
        if traceback_info:
            embed.add_embed_field(
                name="Traceback",
                value=f"```{traceback_info[:500]}...```",
                inline=False,
            )

        # Set footer
        embed.set_footer(text=f"{self.project_name} - Training Failed")

        webhook.add_embed(embed)
        response = webhook.execute()
        return response

    def send_training_start(
        self,
        total_epochs: int,
        batch_size: int,
        train_size: int,
        val_size: int,
        training_on: str,
        number_of_gpus: int,
        learning_rate: float,
    ) -> dict:
        """
        Send training start notification to Discord.
        Args:
            total_epochs: Total number of epochs
            batch_size: Training batch size
            train_size: Size of the training dataset
            val_size: Size of the validation dataset
            training_on: Device used for training (e.g., CPU, GPU)
            number_of_gpus: Number of GPUs used
            learning_rate: Initial learning rate
        Returns:
            dict: Response from Discord API
        """
        if self.silent_mode: return {"status": "silent_mode_enabled"}

        webhook = self._create_webhook()

        # Create start embed
        embed = self._create_base_embed(
            title="🏁 Training Started",
            description=f"Model training has started for **{total_epochs} epochs**!",
            color="2ecc71",  # Green for start/go
        )

        # Add training details
        embed.add_embed_field(name="Total Epochs", value=f"{total_epochs}", inline=True)
        embed.add_embed_field(name="Batch Size", value=f"{batch_size}", inline=True)
        embed.add_embed_field(
            name="Training Set Size", value=f"{train_size}", inline=True
        )
        embed.add_embed_field(
            name="Validation Set Size", value=f"{val_size}", inline=True
        )
        embed.add_embed_field(name="Training On", value=f"{training_on}", inline=True)
        embed.add_embed_field(
            name="Number of GPUs", value=f"{number_of_gpus}", inline=True
        )
        embed.add_embed_field(
            name="Learning Rate", value=f"{learning_rate}", inline=True
        )

        # Set footer
        embed.set_footer(text=f"{self.project_name} - Training Started")

        webhook.add_embed(embed)
        response = webhook.execute()
        return response

    def send_training_completion(
        self,
        total_epochs: int,
        final_loss: float,
        best_loss: float,
        training_time: str,
        final_loss_curve_path: Optional[Union[str, Path]] = None,
        best_model_path: Optional[Union[str, Path]] = None,
    ) -> dict:
        """
        Send training completion notification to Discord.

        Args:
            total_epochs: Total number of epochs completed
            final_loss: Final training loss
            best_loss: Best loss achieved during training
            training_time: Total training time
            final_loss_curve_path: Path to final loss curve image
            best_model_path: Path to saved best model

        Returns:
            dict: Response from Discord API
        """
        if self.silent_mode: return {"status": "silent_mode_enabled"}

        webhook = self._create_webhook()

        # Create completion embed
        embed = self._create_base_embed(
            title="🎉 Training Completed Successfully!",
            description=f"Model training has finished after **{total_epochs} epochs**!",
            color="2ecc71",  # Green for success
        )

        # Add final loss curve if available
        if final_loss_curve_path and Path(final_loss_curve_path).exists():
            with open(Path(final_loss_curve_path), "rb") as f:
                webhook.add_file(file=f.read(), filename="final_loss_curve.png")
            embed.set_image(url="attachment://final_loss_curve.png")

        # Final results
        embed.add_embed_field(
            name="Final Loss",
            value=f"{final_loss:.6f}",
            inline=True,
        )

        embed.add_embed_field(
            name="Best Loss",
            value=f"{best_loss:.6f}",
            inline=True,
        )

        # Training time
        embed.add_embed_field(
            name="Training Time", value=f"{training_time}", inline=True
        )

        # Model info
        if best_model_path:
            embed.add_embed_field(
                name="Model Saved",
                value=f"Best model saved to: `{Path(best_model_path).name}`",
            )

        # Set footer
        embed.set_footer(text=f"{self.project_name} - Training Complete")

        webhook.add_embed(embed)
        response = webhook.execute()
        return response

    def send_custom_notification(
        self,
        title: str,
        description: str,
        color: str = "95a5a6",
        fields: Optional[list] = None,
        image_path: Optional[Union[str, Path]] = None,
        thumbnail_path: Optional[Union[str, Path]] = None,
    ) -> dict:
        """
        Send a custom notification to Discord.

        Args:
            title: Notification title
            description: Notification description
            color: Embed color (hex string)
            fields: List of field dictionaries with 'name', 'value', and optional 'inline' keys
            image_path: Path to main image
            thumbnail_path: Path to thumbnail image

        Returns:
            dict: Response from Discord API
        """
        if self.silent_mode: return {"status": "silent_mode_enabled"}
        
        webhook = self._create_webhook()

        embed = self._create_base_embed(title, description, color)

        # Add custom fields
        if fields:
            for field in fields:
                embed.add_embed_field(
                    name=field.get("name", "Field"),
                    value=field.get("value", "Value"),
                    inline=field.get("inline", False),
                )

        # Add images
        self._add_image_files(webhook, embed, image_path, thumbnail_path)

        # Set footer
        embed.set_footer(text=f"{self.project_name}")

        webhook.add_embed(embed)
        response = webhook.execute()
        return response
