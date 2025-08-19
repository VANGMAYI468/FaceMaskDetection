import pygame
import time

pygame.mixer.init()
pygame.mixer.music.load("assets/alert.wav")
pygame.mixer.music.set_volume(1.0)
pygame.mixer.music.play()

time.sleep(5)  # Wait while sound plays


