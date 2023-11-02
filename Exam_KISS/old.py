if landmarkSeen:
                    landmarkIndex = 0
                    for i in range(len(seenLandmarks)):
                        if seenLandmarks[i].id == goalID:
                            landmarkIndex = i
                
                    # Robotten kører og apporacher landmarket
                    landmarkFound, maxdist = drive_carefully_to_landmark(seenLandmarks[landmarkIndex], frontLimit, sideLimit)
               