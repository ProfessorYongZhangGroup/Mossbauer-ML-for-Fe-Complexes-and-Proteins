from enum import auto
from typing import Sequence

import numpy as np
from rdkit.Chem.rdchem import Atom, HybridizationType, ChiralType

from chemprop.featurizers.base import VectorFeaturizer
from chemprop.utils.utils import EnumMapping


class TMCAtomFeaturizer(VectorFeaturizer[Atom]):
    """A :class:`MultiHotAtomFeaturizer` uses a multi-hot encoding to featurize atoms.

    .. seealso::
        The class provides three default parameterization schemes:

        * :meth:`MultiHotAtomFeaturizer.v1`
        * :meth:`MultiHotAtomFeaturizer.v2`
        * :meth:`MultiHotAtomFeaturizer.organic`

    The generated atom features are ordered as follows:
    * atomic number
    * degree
    * formal charge
    * chiral tag
    * number of hydrogens
    * hybridization
    * aromaticity
    * mass

    .. important::
        Each feature, except for aromaticity and mass, includes a pad for unknown values.

    Parameters
    ----------
    atomic_nums : Sequence[int]
        the choices for atom type denoted by atomic number. Ex: ``[4, 5, 6]`` for C, N and O.
    degrees : Sequence[int]
        the choices for number of bonds an atom is engaged in.
    formal_charges : Sequence[int]
        the choices for integer electronic charge assigned to an atom.
    chiral_tags : Sequence[int]
        the choices for an atom's chiral tag. See :class:`rdkit.Chem.rdchem.ChiralType` for possible integer values.
    num_Hs : Sequence[int]
        the choices for number of bonded hydrogen atoms.
    hybridizations : Sequence[int]
        the choices for an atom’s hybridization type. See :class:`rdkit.Chem.rdchem.HybridizationType` for possible integer values.
    """


    def __init__(
        self,
        atomic_nums: Sequence[int],
        degrees: Sequence[int],
        formal_charges: Sequence[int],
        num_Hs: Sequence[int],
        hybridizations: Sequence[int],
        chirality: Sequence[int],
        mask: Sequence[bool],
    ):
        self.atomic_nums = {j: i for i, j in enumerate(atomic_nums)}
        self.degrees = {i: i for i in degrees}
        self.formal_charges = {j: i for i, j in enumerate(formal_charges)}
        self.num_Hs = {i: i for i in num_Hs}
        self.hybridizations = {ht: i for i, ht in enumerate(hybridizations)}
        self.chirality = {ct: i for i, ct in enumerate(chirality)}
        self.mask = mask
        
        self.__electronegativity = {
            "H" :2.20 ,"Li":0.98,"Be":1.57,"B" :2.04,"C" :2.55,"N" :3.04,"O" :3.44,
            "F" :3.98,"Na":0.93,"Mg":1.31,"Al":1.61,"Si":1.90,"P" :2.19,"S" :2.58,
            "Cl":3.16,"K" :0.82,"Ca":1.00,"Sc":1.36,"Ti":1.54,"V" :1.63,"Cr":1.66,
            "Mn":1.55,"Fe":1.83,"Co":1.88,"Ni":1.91,"Cu":1.90,"Zn":1.65,"Ga":1.81,
            "Ge":2.01,"As":2.18,"Se":2.55,"Br":2.96,"Kr":3.00,"Rb":0.82,"Sr":0.95,
            "Y" :1.22,"Zr":1.33,"Nb":1.6 ,"Mo":2.16,"Tc":1.9 ,"Ru":2.2 ,"Rh":2.28,
            "Pd":2.20,"Ag":1.93,"Cd":1.69,"In":1.78,"Sn":1.96,"Sb":2.05,"Te":2.1 ,
            "I" :2.66,"Pt":2.28}

        self._subfeats: list[dict] = [
            self.atomic_nums,
            self.degrees,
            self.formal_charges,
            self.num_Hs,
            self.hybridizations,
            self.chirality,
        ]
        subfeat_sizes = [
            1 + len(self.atomic_nums),
            1 + len(self.degrees),
            1 + len(self.num_Hs),
            1 + len(self.hybridizations),
            1 + len(self.chirality),
            1,
            1,
            1,
        ]
        self.__size = sum(subfeat_sizes)

    def __len__(self) -> int:
        return self.__size

    def __call__(self, a: Atom | None) -> np.ndarray:
        x = np.zeros(self.__size)

        if a is None:
            return x

        feats = [
            a.GetAtomicNum(),
            a.GetTotalDegree(),
            int(a.GetTotalNumHs()),
            a.GetHybridization(),
            a.GetChiralTag(),
        ]
        i = 0
        for k, (feat, choices) in enumerate(zip(feats, self._subfeats)):
            j = choices.get(feat, len(choices))
            x[i + j] = 1 if self.mask[k] else 0
            i += len(choices) + 1
        x[i] = int(a.GetIsAromatic()) if self.mask[5] else 0
        x[i + 1] = 0.01 * a.GetMass() if self.mask[6] else 0
        x[i + 2] = 0.25 * self.__electronegativity[a.GetSymbol()] if self.mask[7] else 0

        return x

    @classmethod
    def complex(cls,mask=None):
        """An implementation that includes an atom type bit for all elements in the first four rows of the periodic table plus iodine."""
        if mask is None:
            mask = [True for x in range(8)]
        return cls(
            atomic_nums=list(range(1, 37)) + [53, 78],
            degrees=list(range(6)),
            formal_charges=[-1, -2, 1, 2, 0],
            num_Hs=list(range(5)),
            hybridizations=[
                HybridizationType.S,
                HybridizationType.SP,
                HybridizationType.SP2,
                HybridizationType.SP2D,
                HybridizationType.SP3,
                HybridizationType.SP3D,
                HybridizationType.SP3D2,
            ],
            chirality=[
                ChiralType.CHI_TETRAHEDRAL_CW,
                ChiralType.CHI_TETRAHEDRAL_CCW,
                ChiralType.CHI_SQUAREPLANAR,
                ChiralType.CHI_TRIGONALBIPYRAMIDAL,
                ChiralType.CHI_OCTAHEDRAL,
            ],
            mask=mask
        )
