from typing import List
import cv2
import matplotlib.pyplot as plt
import numpy as np

PURPLE = 'p'
SKIN_TONE = 'x'
RED = 'r'
BLUE = 'b'
GREEN = 'g'
YELLOW = 'y'
BLACK = 'k'
GRAY = 'a'
PINK = 'i'

COLOR_CHAR_TO_BGR = {
    # PURPLE:    (255,   0, 255),
    PURPLE:    (128, 0, 128),
    SKIN_TONE: (203, 192, 255),
    RED:       (0,     0, 255),
    BLUE:      (255,   0,   0),
    GREEN:     (0,   255,   0),
    YELLOW:    (0,   255, 255),
    BLACK:     (0,     0,   0),
    GRAY:      (128, 128, 128),
    PINK:      (147, 20, 255)
}

# most common rgb color for purple is (128, 0, 128)
# most common rgb color for skin tone is (255, 224, 189)
# most common rgb color for pink is (203, 192, 255),


class Screw():
    def __init__(self, nuts: list[str], max_amount: int) -> None:
        """
        Class representing a screw with nuts.

        Args:
            nuts (list[str]): List of nuts on the screw, represented by their color characters. The last element is the topmost nut.
            max_amount (int): Maximum number of nuts that can be placed on the screw.
        """

        assert len(nuts)<=max_amount, f"ERROR: Got {nuts}"

        self.nuts=nuts
        self.max_amount=max_amount
    
    def get_upper_nuts(self) -> str:
        """Returns the upper nuts whose color matches the upper nut color"""
        if self.is_empty() or self.is_completed():
            return None
        
        upper_nut=self.nuts[-1]
        upper_nuts=upper_nut

        for k in range(len(self.nuts)-1):
            if self.nuts[-2-k]==upper_nut:
                upper_nuts+=upper_nut
            else:
                return upper_nuts
    
        # if not len(self.nuts)==1:
        #     print(f"get_upper_nuts(): WARNING: this function should never reach here. probably a bug. {self.nuts=}, {upper_nut=}, {upper_nuts=}")
        #     exit()

        return upper_nuts

    def is_full(self) -> bool:
        return len(self.nuts)==self.max_amount
    
    def is_empty(self) -> bool:
        return len(self.nuts)==0

    def can_insert_nuts(self, nuts: str):

        assert len(set(nuts))==1, f"Trying to insert nuts with different color {nuts=}"

        if self.is_full():
            return False
        
        if len(self.nuts)+len(nuts)>self.max_amount:
            return False
        
        if self.is_empty():
            return True

        return nuts[0]==self.nuts[-1]

    def pop_upper_nuts(self, amount: int = None) -> str:
        upper_nuts=self.get_upper_nuts()

        if upper_nuts is None:
            return None
        
        if amount is not None:
            upper_nuts=upper_nuts[:amount]
        
        self.nuts=self.nuts[:-len(upper_nuts)]

        return upper_nuts
    
    def insert_nuts(self, nuts: str) -> bool:
        if not self.can_insert_nuts(nuts):
            return False
        
        self.nuts+=nuts
        return True
    
    def is_completed(self) -> bool:
        if self.is_empty():
            return True
        return self.is_full() and len(set(self.nuts))==1
    
    def __repr__(self) -> str:
        return str(self.nuts)

class Move():
    def __init__(self, source_screw_id: int, destination_screw_id: int, nuts_color: str, nuts_amount: int) -> None:
        self.source_screw_id      = source_screw_id
        self.destination_screw_id = destination_screw_id
        self.nuts_color           = nuts_color
        self.nuts_amount          = nuts_amount
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(source={self.source_screw_id}, destination={self.destination_screw_id}, color={self.nuts_color}, amount={self.nuts_amount})"

    def __str__(self) -> str:
        return self.__repr__()
        return f"{self.source_screw_id}{self.destination_screw_id}{self.nuts_color}{self.nuts_amount}"

class Scenario():
    def __init__(self, screws: List[Screw]) -> None:
        self.screws=screws
        self.moves_history=[]
    
    def get_possible_moves(self) -> List[Move]:
        moves=[]
        for i in range(len(self.screws)):
            for j in range(len(self.screws)):
                if i==j: continue
                
                upper_nuts=self.screws[i].get_upper_nuts()
                
                if upper_nuts is None:
                    continue
                
                if self.screws[j].can_insert_nuts(upper_nuts):
                    move=Move(
                        source_screw_id=i,
                        destination_screw_id=j,
                        nuts_amount=len(upper_nuts),
                        nuts_color=upper_nuts[0]
                    )
                    moves.append(move)
        
        return moves
    
    def get_possible_smart_moves(self) -> List[Move]:
        """Only moves that bring us closer to the solution. For example, if a screw has only nuts of one color and there is an empty screw, do not move nuts to the empty screw."""
        moves=[]
        for i in range(len(self.screws)):
            for j in range(len(self.screws)):
                if i==j: continue
                
                upper_nuts=self.screws[i].get_upper_nuts()
                
                if upper_nuts is None:
                    continue

                if len(upper_nuts)==len(self.screws[i].nuts):
                    # all nuts on screw are of the same color
                    if self.screws[j].is_empty():
                        # do not move to empty screw
                        continue
                
                if self.screws[j].can_insert_nuts(upper_nuts):
                    move=Move(
                        source_screw_id=i,
                        destination_screw_id=j,
                        nuts_amount=len(upper_nuts),
                        nuts_color=upper_nuts[0]
                    )
                    moves.append(move)
        
        return moves
    
    def perform_move(self, move: Move):
        nuts=self.screws[move.source_screw_id].pop_upper_nuts()
        assert len(nuts)==move.nuts_amount
        success=self.screws[move.destination_screw_id].insert_nuts(nuts)
        # print(f"Performing move: {move}, popped nuts: {nuts}. Success: {success}")

        if not success:
            print(f"Cannot perform move {move}")
            return False
        
        self.moves_history.append(move)

        return True
        
    def undo_move(self, move: Move) -> bool:
        try:
            nuts=self.screws[move.destination_screw_id].nuts[-move.nuts_amount:]
            self.screws[move.destination_screw_id].nuts=self.screws[move.destination_screw_id].nuts[:-move.nuts_amount]
            self.screws[move.source_screw_id].nuts+=nuts # force insert despite possible color difference
            self.moves_history.pop()
            # print(f"Undid move: {move}, moved back nuts: {nuts}")
            return True
        except Exception as e:
            print("undo_move(): ",e)
            print(f"Cannot undo move {move}")
            self.print()
            # exit()
            assert False
        return False
    
    def is_completed(self) -> bool:
        for screw in self.screws:
            if not screw.is_completed():
                return False
        
        return True
    
    def print(self):
        max_screw_size=max([screw.max_amount for screw in self.screws])

        for level in list(range(max_screw_size))[::-1]:
            for screw in self.screws:
                if level>screw.max_amount:
                    print(" ", end='\t')
                if level>=len(screw.nuts):
                    print("|", end='\t')
                else:
                    print(screw.nuts[level], end='\t')
            print()
    
    def display(self):
        img_height=200
        img_width=100*len(self.screws)
        screw_width=80
        screw_spacing=20

        image=np.ones((img_height, img_width, 3), dtype=np.uint8)*255

        for screw_id, screw in enumerate(self.screws):
            x_start=screw_id*100 + screw_spacing//2
            x_end=x_start + screw_width

            for nut_level in range(screw.max_amount):
                y_end=img_height - (nut_level * (img_height//screw.max_amount))
                y_start=y_end - (img_height//screw.max_amount)

                if nut_level<len(screw.nuts):
                    color_char=screw.nuts[nut_level]
                    color_bgr=COLOR_CHAR_TO_BGR[color_char]
                    cv2.rectangle(image, (x_start, y_start), (x_end, y_end), color_bgr, -1)
                else:
                    cv2.rectangle(image, (x_start, y_start), (x_end, y_end), (200,200,200), 1)
        
        cv2.imshow("Scenario", image)
        key = cv2.waitKey(0) & 0xFF
        if key == ord('q'):
            # close all windows and exit
            cv2.destroyAllWindows()
            exit()

    def display2(self):

        img_height = 400
        screw_height = img_height - 100
        img_width = 100 * len(self.screws)
        screw_width = 80
        screw_spacing = 20

        image = np.ones((img_height, img_width, 3), dtype=np.uint8) * 255

        for screw_id, screw in enumerate(self.screws):
            x_start = screw_id * 100 + screw_spacing // 2
            x_end = x_start + screw_width

            for nut_level in range(screw.max_amount):
                y_end = screw_height - (nut_level * (screw_height // screw.max_amount))
                y_start = y_end - (screw_height // screw.max_amount)

                if nut_level < len(screw.nuts):
                    color_char = screw.nuts[nut_level]
                    color_bgr = COLOR_CHAR_TO_BGR[color_char]
                    cv2.rectangle(image, (x_start, y_start), (x_end, y_end), color_bgr, -1)
                else:
                    cv2.rectangle(image, (x_start, y_start), (x_end, y_end), (200, 200, 200), 1)
            
            cv2.putText(image, f"{screw_id}", (x_start-5, screw_height + 80), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 1) # here the font size is 0.5 and thickness is 1

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # ----------------------------------
        # Crear figura SOLO una vez
        # ----------------------------------
        if not hasattr(self, "fig"):
            plt.ion()  # modo interactivo
            self.fig, self.axes = plt.subplots(1, 3, figsize=(10, 5))
            # self.fig.show(block=False)
            plt.show(block=False)
        else:
            # Limpiar ejes existentes
            for ax in self.axes:
                ax.clear()

        # ----------------------------------
        # Imagen
        # ----------------------------------
        self.axes[0].imshow(image)
        self.axes[0].axis("off")
        self.axes[0].set_title(f"Current Scenario. Completed: {self.is_completed()}")

        # ----------------------------------
        # Texto
        # ----------------------------------
        self.axes[1].axis("off")
        text = "\n".join([str(move) for move in self.moves_history])
        self.axes[1].text(
            0, 1, text,
            fontsize=12,
            verticalalignment="top"
        )
        self.axes[1].set_title(f"Move history ({len(self.moves_history)})")

        self.axes[2].axis("off")
        possible_moves = self.get_possible_smart_moves()
        text = "\n".join([str(move) for move in possible_moves])
        self.axes[2].text(
            0, 1, text,
            fontsize=12,
            verticalalignment="top"
        )
        self.axes[2].set_title("Possible moves")

        self.fig.tight_layout()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.waitforbuttonpress()



MAX_DEPTH=30
amount_scenarios_explored=0
amount_max_depth_reached=0
def explore_scenario(scenario: Scenario, current_depth: int = 0, display: bool = False) -> bool:

    # display
    if display:
        scenario.display2()

    global amount_scenarios_explored
    global amount_max_depth_reached
    amount_scenarios_explored+=1

    if current_depth<15:
        print(f"Exploring depth {current_depth}, scenarios explored: {amount_scenarios_explored}, max depth reached: {amount_max_depth_reached}", end='\r')

    if current_depth>MAX_DEPTH:
        amount_max_depth_reached+=1
        # print(f"Max depth reached {amount_max_depth_reached}", end='\r')
        return False

    if scenario.is_completed():
        print("COMPLETED")
        scenario.print()
        return True

    moves=scenario.get_possible_smart_moves()

    for move in moves:
        
        success=scenario.perform_move(move)
        if not success:
            print(f"Failed to perform move {move} during exploration")
            exit()
        
        completed=explore_scenario(scenario, current_depth+1)

        if completed:
            print(f"Completed with move {move}")
            # exit()
            return True
        
        success=scenario.undo_move(move)
        if not success:
            print(f"Failed to undo move {move} during exploration")
            exit()

    return False
        

if __name__=='__main__':
    s=Scenario(
        screws=[
            Screw(list('brb'), max_amount=3),
            Screw(list('brx'), max_amount=3),
            Screw(list('yyg'), max_amount=3),
            Screw(list('pxg'), max_amount=3),
            Screw(list('ygp'), max_amount=3),
            Screw(list('xpr'), max_amount=3),
            Screw(list(''), max_amount=3),
        ]
    )

    

    s=Scenario(
        screws=[
            Screw([PURPLE, SKIN_TONE, YELLOW, GREEN][::-1], max_amount=4),
            Screw([GREEN, BLACK, GRAY, BLUE][::-1], max_amount=4),
            Screw([RED, YELLOW, YELLOW, PINK][::-1], max_amount=4),
            Screw([BLUE, BLACK, SKIN_TONE, PURPLE][::-1], max_amount=4),
            Screw([BLACK, YELLOW, BLACK, PINK][::-1], max_amount=4),
            Screw([PURPLE, GRAY, GRAY, RED][::-1], max_amount=4),
            Screw([SKIN_TONE, RED, GREEN, BLUE][::-1], max_amount=4),
            Screw([SKIN_TONE, PINK, PURPLE, RED][::-1], max_amount=4),
            Screw([GRAY, GREEN, PINK, BLUE][::-1], max_amount=4),
            Screw(list(''), max_amount=4),
            Screw(list(''), max_amount=4),
        ]
    )

    # s=Scenario(
    #     screws=[
    #         Screw([BLUE, BLUE, YELLOW, GREEN, YELLOW, BLUE, BLUE, BLUE], max_amount=8),
    #         Screw([RED, GREEN, GREEN, GREEN, BLUE, GREEN, YELLOW, YELLOW], max_amount=8),
    #         Screw([RED, BLUE, GREEN, GREEN, RED, RED, RED, RED], max_amount=8),
    #         Screw([RED, GREEN, YELLOW, RED, YELLOW, BLUE, YELLOW, YELLOW], max_amount=8),
    #         Screw(list(''), max_amount=8),
    #         Screw(list(''), max_amount=8),
    #     ]
    # )

    

    # s=Scenario(
    #     screws=[
    #         Screw(list('brb'), max_amount=3),
    #         Screw(list('brr'), max_amount=3),
    #         Screw(list(''), max_amount=3),
    #     ]
    # )

    s.print()
    # s.display2()
    explore_scenario(s, display=False)
    s.print()
    print(amount_scenarios_explored)


    