
# funzione aggiunta in rigid_entity.py
def add_joint(self, name, type, eq_obj1id, eq_obj2id, eq_data, sol_params):
        if type == gs.EQUALITY_TYPE.CONNECT:
            eq_obj1id += self._link_start
            eq_obj2id += self._link_start
        elif type == gs.EQUALITY_TYPE.JOINT:
            eq_obj1id += self._joint_start
            eq_obj2id += self._joint_start
        elif type == gs.EQUALITY_TYPE.WELD:
            eq_obj1id += self._link_start
            eq_obj2id += self._link_start
        else:
            gs.logger.warning(f"Equality type {type} not supported. Only CONNECT, JOINT, and WELD are supported.")
        equality = RigidEquality(
            entity=self,
            name=name,
            idx=self.n_equalities + self._equality_start,
            type=type,
            eq_obj1id=eq_obj1id,
            eq_obj2id=eq_obj2id,
            eq_data=eq_data,
            sol_params=sol_params,
        )
        self._equalities.append(equality)
        return equality