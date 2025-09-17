from typing import List

class Screw():
    def __init__(self, nuts: str, max_amount: int) -> None:

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
        return f"{self.source_screw_id}{self.destination_screw_id}{self.nuts_color}{self.nuts_amount}"

class Scenario():
    def __init__(self, screws: List[Screw]) -> None:
        self.screws=screws
    
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
    
    def perform_move(self, move: Move):
        nuts=self.screws[move.source_screw_id].pop_upper_nuts()
        assert len(nuts)==move.nuts_amount
        success=self.screws[move.destination_screw_id].insert_nuts(nuts)

        if not success:
            print(f"Cannot perform move {move}")
        
        return success
        
        
    def undo_move(self, move: Move) -> bool:
        try:
            nuts=self.screws[move.destination_screw_id].nuts[-move.nuts_amount:]
            self.screws[move.destination_screw_id].nuts=self.screws[move.destination_screw_id].nuts[:-move.nuts_amount]
            self.screws[move.source_screw_id].nuts+=nuts # force insert despite possible color difference
        except Exception as e:
            print("undo_move(): ",e)
            print(f"Cannot undo move {move}")
            self.print()
            # exit()
            assert False
    
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



MAX_DEPTH=20
amount_scenarios_explored=0
def explore_scenario(scenario: Scenario, current_depth: int = 0):

    global amount_scenarios_explored
    amount_scenarios_explored+=1

    if current_depth>MAX_DEPTH:
        print("Max depth reached")
        return False

    if scenario.is_completed():
        print("COMPLETED")
        scenario.print()
        return True

    moves=scenario.get_possible_moves()

    for move in moves:
        success=scenario.perform_move(move)
        if not success:
            exit()
        completed=explore_scenario(scenario, current_depth+1)
        scenario.undo_move(move)
        if completed:
            print(f"Completed with move {move}")
            return True
    
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

    

    # s=Scenario(
    #     screws=[
    #         Screw(list('brb'), max_amount=3),
    #         Screw(list('brr'), max_amount=3),
    #         Screw(list(''), max_amount=3),
    #     ]
    # )

    s.print()
    explore_scenario(s)
    s.print()
    print(amount_scenarios_explored)


    