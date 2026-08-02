# ── Helper: Live Plotting Function ──
def plot_loss_curves(log_df, save_path):
    if log_df.empty:
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Total loss
    axes[0, 0].plot(log_df['epoch'], log_df['train_loss'], label='Train Loss', marker='o')
    axes[0, 0].plot(log_df['epoch'], log_df['val_loss'], label='Val Loss', marker='x')
    axes[0, 0].set_title('Total Loss')
    axes[0, 0].legend()
    
    # CI loss
    axes[0, 1].plot(log_df['epoch'], log_df['train_loss_ci'], label='Train CI', marker='o')
    axes[0, 1].plot(log_df['epoch'], log_df['val_loss_ci'], label='Val CI', marker='x')
    axes[0, 1].set_title('Cell Indicator Loss')
    axes[0, 1].legend()
    
    # Tracking Losses
    axes[1, 0].plot(log_df['epoch'], log_df['train_loss_la'], label='Train LA (Full)', marker='o')
    axes[1, 0].plot(log_df['epoch'], log_df['train_loss_lb'], label='Train LB (Maxima)', marker='o')
    if 'val_loss_la' in log_df.columns:
        axes[1, 0].plot(log_df['epoch'], log_df['val_loss_la'], label='Val LA', marker='x')
        axes[1, 0].plot(log_df['epoch'], log_df['val_loss_lb'], label='Val LB', marker='x')
    axes[1, 0].set_title('Tracking Losses')
    axes[1, 0].legend()
    
    # Alpha
    axes[1, 1].plot(log_df['epoch'], log_df['train_alpha'], label='Curriculum Alpha', marker='o')
    axes[1, 1].set_title('Curriculum Alpha')
    axes[1, 1].legend()
    
    fig.tight_layout()  
    
    plt.savefig(save_path)
    display(fig)
    plt.close()
