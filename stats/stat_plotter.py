import numpy as np
import matplotlib.pyplot as plt
from models import CombatAgent
from stats.discord_webhook import DiscordWebhook

def plot_stat(train_scores: list, all_scores:list, train_quartiles: list, 
              epoch: int, agent: CombatAgent, bot: DiscordWebhook, epoch_start: int=0, plot: bool=True) -> str:
    """Helper function for saving and plotting stats locally, these can later be sent via discord

    Args:
        train_scores (list): training scores for current epoch
        all_scores (list): training scores in index 0, training kills in index 1
        train_quartiles (list): epoch quartiles for training scores
        epoch (int): epoch index starting at 0
        agent (CombatAgent): the RL agent
        bot (discord_bot): the discord bot instance
        epoch_start (int, optional): legacy option that does nothing. Defaults to 0.
        plot (bool, optional): whether to plot the stats. Defaults to True.

    Returns:
        str: a sentence describing the stats
    """    
    
    train_scores = np.array(train_scores)
    Q1, Q2, Q3 = np.percentile(train_scores, 25), np.median(train_scores), np.percentile(train_scores, 75)
    mean = train_scores.mean()
    
    train_quartiles[0].append(Q1)
    train_quartiles[1].append(Q2)
    train_quartiles[2].append(Q3)
    train_quartiles[3].append(mean)
    
    if plot:
        plt.clf()
        
        plt.bar(np.arange(len(train_scores)), train_scores, 1)
        plt.title(f"Epoch {epoch+1}, eps = {agent.eps}")
        plt.xlim(0, len(train_scores)+1)
        plt.xlabel("episode")
        plt.ylabel("score")
        plt.savefig(f"{bot.path}/{epoch}.png")
        plt.clf()
        
        plt.bar(np.arange(len(all_scores[0])), all_scores[0], 1)
        plt.title("All epoches")
        plt.xlim(0, len(all_scores[0])+1)
        plt.xlabel("episode")
        plt.ylabel("score")
        plt.savefig(f"{bot.path}/{epoch}a.png")
        plt.clf()
        
        episode_x = np.arange(len(all_scores[0]))
        
        plt.plot(episode_x, np.cumsum(all_scores[0])/episode_x+1, 'k-', label="Mean")
        plt.xlabel("episode")
        plt.ylabel("mean score")
        plt.title("Mean score")
        plt.legend()
        plt.xlim(0, len(episode_x))
        plt.savefig(f"{bot.path}/train_quartiles.png")
        plt.clf()
        
        kd_ratio = np.cumsum(all_scores[1])/(episode_x+1)
        kd_ratio_max = kd_ratio.max()
        plt.plot(episode_x, kd_ratio)
        plt.title("Average K/D Score")
        plt.xlabel("episode")
        plt.ylabel("kill/death ratio")
        plt.xlim(0, len(all_scores[1])+1)
        plt.ylim(-1, kd_ratio_max+1)
        plt.savefig(f"{bot.path}/train_kill_counts.png")
        plt.clf()
        
        plt.subplot(2, 2, 1)
        plt.bar(np.arange(len(train_scores)), train_scores, 1)
        plt.title(f"Epoch {epoch+1}, eps = {agent.eps}")
        plt.xlim(0, len(train_scores)+1)
        plt.xlabel("episode")
        plt.ylabel("score")
        
        plt.subplot(2, 2, 2)
        plt.bar(np.arange(len(all_scores[0])), all_scores[0], 1)
        plt.title("All epoches")
        plt.xlim(0, len(all_scores[0])+1)
        plt.xlabel("episode")
        plt.ylabel("score")
        
        plt.subplot(2, 2, 3)
        plt.xlim(0, len(train_quartiles[0])+1)
        plt.plot(episode_x, np.cumsum(all_scores[0])/episode_x+1, 'k-', label="Mean")
        plt.xlabel("episode")
        plt.ylabel("mean score")
        plt.title("Mean score")
        plt.xlim(0, len(episode_x))
        
        plt.subplot(2, 2, 4)
        plt.plot(episode_x, kd_ratio)
        plt.title("Average K/D Score")
        plt.xlabel("episode")
        plt.ylabel("kill/death ratio")
        plt.xlim(0, len(all_scores[1])+1)
        plt.ylim(-1, kd_ratio_max+1)
        
        plt.savefig(f"{bot.path}/current.png")
        plt.clf()
    
    np.save(f"{bot.path}/train_kill_counts.npy", np.asfarray(all_scores[1]))
    np.save(f"{bot.path}/scores_{epoch}.npy", train_scores)
    
    stats = f"Result:\nmean: {mean:.2f} +- {train_scores.std():.2f}\nQ1: {Q1}\nQ2: {Q2}\nQ3: {Q3}\nmin: {train_scores.min():.2f}\nmax: {train_scores.max():.2f}"
    print(stats)
    return stats