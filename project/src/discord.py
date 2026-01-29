from notifications import DiscordNotifier
import datetime


def main():

    # Initialize the notifier with your webhook URL
    notifier = DiscordNotifier(
        webhook_url="https://discord.com/api/webhooks/1466392738609238046/YOGa8j4HL9wKYeQXXyFdIR_j-vxs5jGYYekNnY0YSlBy-0aJnFwHXMfGPNxxLkMh5FE-",
        project_name="BIT Thesis Project",
        project_url="https://github.com/matejkrenek/BIT-thesis",
        avatar_name="PCN Training Bot",
    )

    response = notifier.send_training_completion(
        total_epochs=100,
        final_loss=0.0234,
        best_loss=0.0198,
        training_time=datetime.timedelta(hours=5, minutes=32, seconds=15),
        final_loss_curve_path="./final_loss_curve.png",
        best_model_path="./checkpoints/pcn_v2_best.pth",
    )


if __name__ == "__main__":
    main()
