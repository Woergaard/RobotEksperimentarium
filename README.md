# Robot Eksperimentarium
Repo til faget Robot Eksperimentarium på Københavns Universitet (2023).


Tanker til at løse vores opgaver: 

Exercise 2: 
Opgave 1: 
- sætte en grænse værdi til væg (200 millimeter)
- Tjek de mulige veje den kan tage (højre, venstre, bagud) 
- Sensor værdi er i millimeter. 


| Foran | Højre | Venstre | Bag | Kommando |
|-------|-------|---------|-----|----------|
|   :white_check_mark:    |  :white_check_mark:     |    :white_check_mark:     |   :white_check_mark:  |    Kør frem      |
|   :x:    |  :white_check_mark:     |    :white_check_mark:     |   :white_check_mark:  |   Kør til højre eller venstre     |
|   :x:    |    :x:   |    :white_check_mark:     |   :white_check_mark:  |    Vend om og kør bagud       |
|   :x:    |    :x:   |     :x:    |  :white_check_mark:   |    Vend om og kør bagud      |
|   :x:    |    :x:   |      :x:   |    :x: |    STOP      |
|     :white_check_mark:  |    :white_check_mark:   |    :white_check_mark:     |  :x:   |   Kør frem og til højre eller venstre     |
|   :white_check_mark:    |    :white_check_mark:   |      :x:   |  :x:   |    Kør fremad/lidt til højre     |
|   :white_check_mark:    |    :x:   |     :x:    |   :x:  |   Kør fremad    |

